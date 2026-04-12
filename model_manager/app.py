"""Application orchestrator — wires all subsystems together."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from model_manager.agent.conversation import ConversationManager
from model_manager.agent.error_agent import ErrorDiagnosisAgent
from model_manager.backends.selector import BackendSelector
from model_manager.catalog.catalog import ModelCatalog
from model_manager.catalog.recommender import ModelRecommender
from model_manager.config.paths import ensure_dirs
from model_manager.config.settings import Settings, get_settings
from model_manager.core.constants import SessionState
from model_manager.core.events import (
    LogLineEvent, StepCompletedEvent, StepFailedEvent, StepStartedEvent, bus,
)
from model_manager.core.exceptions import (
    BranchFailedError, BranchDepthExceededError, APIKeyMissingError,
)
from model_manager.hardware.detector import HardwareDetector
from model_manager.hardware.profile import HardwareProfile
from model_manager.permissions.factory import get_permission_manager
from model_manager.recovery.branch import BranchExecutor
from model_manager.recovery.context import build_error_context
from model_manager.recovery.resume import ResumeCoordinator
from model_manager.state.machine import StateMachine
from model_manager.state.models import InstallationState, InstallStep
from model_manager.state.store import StateStore
from model_manager.storage.planner import StoragePlanner
from model_manager.ui.chat import ChatInput
from model_manager.ui.dashboard import Dashboard


class App:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        resume_session_id: Optional[str] = None,
        force_backend: Optional[str] = None,
        force_path: Optional[str] = None,
        force_model: Optional[str] = None,
    ) -> None:
        self._settings        = settings or get_settings()
        self._resume_id       = resume_session_id
        self._force_backend   = force_backend
        self._force_path      = Path(force_path) if force_path else None
        self._force_model     = force_model

        ensure_dirs()

        self._store            = StateStore()
        self._hardware: Optional[HardwareProfile] = None
        self._catalog          = ModelCatalog()
        self._dashboard        = Dashboard()
        self._chat_input       = ChatInput()
        self._conversation: Optional[ConversationManager] = None
        self._perm             = get_permission_manager()
        self._chat_confirmed   = False   # True when user confirmed model+path via chat
        self._hf_token: Optional[str] = None   # set when user provides HF token

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._dashboard.start()
        try:
            if self._resume_id:
                state = self._store.load(self._resume_id)
            else:
                state = InstallationState()
                self._store.save(state)

            sm = StateMachine(state, self._store)
            await self._main_loop(state, sm)
        except KeyboardInterrupt:
            await self._log("Interrupted by user.")
        except Exception as e:
            await self._log(f"[error]Fatal: {e}[/error]")
            raise
        finally:
            self._dashboard.stop()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _main_loop(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        # Phase 1: detect hardware
        await self._detect_hardware(state, sm)

        # Phase 2: load catalog and let user browse / select model
        await self._browse_catalog(state, sm)
        if sm.current in (SessionState.ABORTED.value, SessionState.FAILED.value):
            return

        # Phase 3: storage analysis
        await self._analyze_storage(state, sm)

        # Phase 4: confirm plan with user
        if not await self._confirm_plan(state, sm):
            return

        # Phase 5: backend selection
        await self._select_backend(state, sm)

        # Phase 6: install (with error recovery loop)
        await self._install_with_recovery(state, sm)

    # ── Phases ────────────────────────────────────────────────────────────────

    async def _detect_hardware(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        await sm.trigger("start")
        await self._log("Detecting hardware...")
        try:
            detector = HardwareDetector()
            self._hardware = await detector.detect()
            state.hardware_profile = self._hardware.to_dict()
            self._store.save(state)
            self._dashboard.set_hardware(self._hardware)
            await sm.trigger("hardware_detected")
        except Exception as e:
            await self._log(f"[error]Hardware detection failed: {e}[/error]")

    async def _browse_catalog(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        # Load catalog
        try:
            self._catalog.load()
            if self._settings.catalog_update_on_start and self._settings.catalog_remote_url:
                await self._catalog.update_from_remote(self._settings.catalog_remote_url)
        except Exception:
            pass   # use bundled catalog

        if self._force_model:
            state.selected_model_id = self._force_model
            self._store.save(state)
            await sm.trigger("model_selected")
            return

        # Build system prompt with hardware + catalog injected
        system_prompt = self._build_recommendation_system_prompt()

        # Try to use Claude for conversation; fall back to simple input
        try:
            self._conversation = ConversationManager()
            # Trigger LLM to open the conversation (not shown to user; LLM reply is first output)
            await self._conversation.stream_response("Hi", system=system_prompt)
        except (APIKeyMissingError, ImportError) as e:
            await self._log(f"[warning]LLM unavailable ({e}) — using simple model selection.[/warning]")
            recs = ModelRecommender(self._catalog).recommend(self._hardware) if self._hardware else []
            rec_text = "\n".join(f"  {i+1}. {r.model.display_name}" for i, r in enumerate(recs))
            await self._simple_model_selection(state, sm, rec_text)
            return

        async def handle_message(text: str) -> None:
            if text.startswith("/"):
                await self._handle_slash(text, state, sm)
                return

            # Extract path from user message BEFORE LLM call so we can store it
            user_path = self._extract_path_from_message(text)
            if user_path:
                state.install_path = user_path
                self._store.save(state)

            response = await self._conversation.stream_response(text, system=system_prompt)

            # Priority: user's explicit text mention > LLM signal
            # This prevents the LLM from picking the wrong variant (e.g. 4B when user said 1B)
            model_entry = self._fuzzy_match_model_from_text(text)
            if model_entry is None:
                model_entry = self._try_parse_install_signal(response)
            if model_entry is None:
                model_entry = self._try_parse_model_selection(text)

            if model_entry:
                state.selected_model_id = model_entry.model_id
                self._chat_confirmed = True   # user confirmed in chat; skip interactive confirm
                self._store.save(state)
                await sm.trigger("model_selected")
                raise _StopLoop()

        try:
            await self._chat_input.run_loop(handle_message)
        except _StopLoop:
            pass

    def _format_hardware_for_llm(self) -> str:
        hw = self._hardware
        if not hw:
            return "Hardware info unavailable."
        lines = [
            f"- OS: {hw.os_platform} {hw.os_version}",
            f"- CPU: {hw.cpu.brand} ({hw.cpu.physical_cores} physical cores, {hw.cpu.architecture})",
            f"- RAM: {hw.ram_total_gb:.1f} GB total, {hw.ram_available_gb:.1f} GB available",
        ]
        for gpu in hw.gpus:
            lines.append(
                f"- GPU: {gpu.name}  {gpu.vram_gb:.1f} GB VRAM  [{gpu.compute_backend.value}]"
                + (f"  CUDA {gpu.cuda_version}" if gpu.cuda_version else "")
            )
        for drive in hw.drives[:4]:
            lines.append(f"- Disk: {drive.path}  {drive.free_gb:.0f}/{drive.total_gb:.0f} GB free")
        return "\n".join(lines)

    def _format_catalog_for_llm(self) -> str:
        lines = []
        for m in self._catalog.all():
            quants = "  |  ".join(
                f"{q.quant_type} {q.file_size_gb:.1f}GB (min_vram={q.min_vram_gb:.1f}GB)"
                for q in m.quantizations
            )
            lines.append(
                f"[{m.model_id}]  {m.display_name}  {m.parameter_count_b:.1f}B params  "
                f"min_ram={m.min_ram_gb:.0f}GB  capabilities={','.join(m.capabilities)}  "
                f"modality={','.join(m.modality)}\n"
                f"  Quantizations: {quants}"
            )
        return "\n\n".join(lines)

    def _build_recommendation_system_prompt(self) -> str:
        from model_manager.agent.base import load_prompt
        base = load_prompt("system_recommendation.txt")
        hw_section = self._format_hardware_for_llm()
        catalog_section = self._format_catalog_for_llm()
        return (
            f"{base}\n\n"
            f"## User's Hardware\n{hw_section}\n\n"
            f"## Available Models\n{catalog_section}"
        )

    async def _simple_model_selection(
        self, state: InstallationState, sm: StateMachine, rec_text: str
    ) -> None:
        from model_manager.ui.console import console
        console.print("\n[header]Recommended models:[/header]")
        console.print(rec_text)
        if self._hardware:
            recs = ModelRecommender(self._catalog).recommend(self._hardware)
        else:
            recs = [type("R", (), {"model": m})() for m in self._catalog.all()[:5]]

        while True:
            raw = await self._chat_input.get_input("Enter model number or ID: ")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(recs):
                    state.selected_model_id = recs[idx].model.model_id
                    self._store.save(state)
                    await sm.trigger("model_selected")
                    return
            except ValueError:
                # Try as model_id
                try:
                    self._catalog.get_by_id(raw)
                    state.selected_model_id = raw
                    self._store.save(state)
                    await sm.trigger("model_selected")
                    return
                except Exception:
                    pass
            await self._log("[warning]Invalid selection, try again.[/warning]")

    async def _analyze_storage(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        await sm.trigger("storage_analyzed")
        model_entry = self._catalog.get_by_id(state.selected_model_id)
        required_gb = model_entry.min_disk_gb

        if self._force_path:
            state.install_path = str(self._force_path)
            self._store.save(state)
            await sm.trigger("plan_ready")
            return

        # Respect path the user already specified in chat
        if state.install_path:
            await self._log(f"Install path: {state.install_path}  (specified by user)")
            await sm.trigger("plan_ready")
            return

        planner = StoragePlanner()
        suggestions = planner.suggest(self._hardware, required_gb, model_entry.display_name)
        best = suggestions[0]
        state.install_path = str(best.path)
        self._store.save(state)
        await self._log(f"Suggested install path: {best.path}  ({best.reason})")
        await sm.trigger("plan_ready")

    async def _confirm_plan(
        self, state: InstallationState, sm: StateMachine
    ) -> bool:
        model_id = state.selected_model_id or "?"
        path     = state.install_path or "?"
        backend  = self._force_backend or "(auto)"

        from model_manager.ui.console import console
        console.print("\n[header]Installation Plan[/header]")
        console.print(f"  Model:   [model]{model_id}[/model]")
        console.print(f"  Path:    {path}")
        console.print(f"  Backend: {backend}")

        # Skip interactive prompt when user already confirmed in the chat conversation
        if self._chat_confirmed:
            console.print("  [muted](Auto-confirmed — user specified model and path in chat)[/muted]")
            await sm.trigger("plan_confirmed")
            return True

        raw = await self._chat_input.get_input("Proceed? [Y/n]: ")
        if raw.lower() in ("", "y", "yes"):
            await sm.trigger("plan_confirmed")
            return True
        await sm.trigger("plan_rejected")
        return False

    async def _select_backend(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        from model_manager.core.constants import InstallBackendType
        pref = None
        if self._force_backend:
            try:
                pref = InstallBackendType(self._force_backend)
            except ValueError:
                pass

        selector = BackendSelector()
        backend = await selector.select(
            self._hardware, user_preference=pref
        )
        state.backend = backend.name
        self._store.save(state)
        await self._log(f"Selected backend: [info]{backend.name}[/info]")
        await sm.trigger("backend_selected")
        await sm.trigger("repos_resolved")  # repo resolution is lightweight for now

    async def _install_with_recovery(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        """Run INSTALLING_DEPENDENCIES → DOWNLOADING_MODEL → VERIFYING_INSTALL with recovery."""
        steps = self._build_steps(state)
        state.steps = steps
        self._store.save(state)

        executor    = BranchExecutor(self._store)
        coordinator = ResumeCoordinator(self._store)
        start_idx   = state.current_step_index

        MAX_USER_RETRIES = 3
        i = start_idx
        user_retry_counts: dict[int, int] = {}   # step_index → number of user-assisted retries

        while i < len(steps):
            step = steps[i]
            await bus.emit(StepStartedEvent(
                step_name=step.description, step_index=i, total_steps=len(steps)
            ))
            start_time = time.monotonic()
            try:
                await self._execute_step(step, state)
                state.current_step_index = i + 1
                self._store.save(state)
                await bus.emit(StepCompletedEvent(step_name=step.description, step_index=i))
                user_retry_counts.pop(i, None)   # reset counter on success
                i += 1
            except Exception as exc:
                await bus.emit(StepFailedEvent(
                    step_name=step.description, step_index=i, error=str(exc)
                ))
                elapsed = time.monotonic() - start_time

                # Error recovery
                try:
                    await sm.trigger("error_captured")
                    ctx = build_error_context(
                        session_id=state.session_id,
                        step_name=step.description,
                        step_index=i,
                        exc=exc,
                        hardware=self._hardware,
                        elapsed_seconds=elapsed,
                    )
                    try:
                        diagnosis_agent = ErrorDiagnosisAgent()
                        diagnosis = await diagnosis_agent.diagnose(ctx)
                    except (APIKeyMissingError, ImportError) as e:
                        await self._log(f"[warning]LLM unavailable ({e}) — manual intervention needed.[/warning]")
                        await sm.trigger("branch_needs_user")
                        return

                    result = await executor.execute(diagnosis, ctx)

                    if result.success:
                        sm.set_resume_target(SessionState.INSTALLING_DEPENDENCIES)
                        await sm.trigger("branch_verified")
                        await coordinator.resume(state, result, i)
                        user_retry_counts.pop(i, None)
                        continue
                    else:
                        await self._log(f"[error]Branch fix failed: {result.error}[/error]")
                        retries = user_retry_counts.get(i, 0)
                        if retries >= MAX_USER_RETRIES:
                            await self._log(
                                f"[error]Step '{step.description}' failed {MAX_USER_RETRIES} times "
                                f"after user intervention. Aborting — please fix the issue and "
                                f"restart with: mm --resume {state.session_id[:8]}[/error]"
                            )
                            await sm.trigger("branch_needs_user")
                            return
                        should_retry = await self._handle_user_intervention(
                            step, diagnosis, state
                        )
                        if should_retry:
                            user_retry_counts[i] = retries + 1
                            sm.set_resume_target(SessionState.INSTALLING_DEPENDENCIES)
                            await sm.trigger("branch_verified")
                            continue
                        await sm.trigger("branch_needs_user")
                        return

                except (BranchDepthExceededError, BranchFailedError) as e:
                    await self._log(f"[error]Recovery exhausted: {e}[/error]")
                    retries = user_retry_counts.get(i, 0)
                    if retries >= MAX_USER_RETRIES:
                        await self._log(
                            f"[error]Step '{step.description}' failed {MAX_USER_RETRIES} times "
                            f"after user intervention. Aborting — please fix the issue and "
                            f"restart with: mm --resume {state.session_id[:8]}[/error]"
                        )
                        await sm.trigger("branch_fatal")
                        return
                    should_retry = await self._handle_user_intervention(
                        step, diagnosis, state
                    )
                    if should_retry:
                        user_retry_counts[i] = retries + 1
                        sm.set_resume_target(SessionState.INSTALLING_DEPENDENCIES)
                        await sm.trigger("branch_verified")
                        continue
                    await sm.trigger("branch_fatal")
                    return

        await sm.trigger("deps_installed")
        await sm.trigger("download_complete")
        await sm.trigger("verified")
        await self._log("[success]Installation complete![/success]")

    def _build_steps(self, state: InstallationState) -> list[InstallStep]:
        model_id = state.selected_model_id or "unknown"
        return [
            InstallStep("mkdir",         "create_directories", f"Create install directory for {model_id}"),
            InstallStep("install_deps",  "install_packages",   "Install huggingface_hub and llama-cpp-python"),
            InstallStep("download_model","download_model",      f"Download {model_id} from HuggingFace"),
            InstallStep("verify",        "verify_install",     "Verify downloaded files"),
            InstallStep("launch_info",   "show_launch_info",   "Show how to launch the model"),
        ]

    async def _execute_step(self, step: InstallStep, state: InstallationState) -> None:
        """Dispatch to the correct handler based on step_type."""
        import sys
        from model_manager.ui.console import console

        if step.step_type == "create_directories":
            install_path = Path(state.install_path or ".")
            model_id = state.selected_model_id or "model"
            # sanitise model_id into a safe folder name
            safe_name = model_id.replace("/", "__").replace("\\", "__")
            model_dir = install_path / safe_name
            model_dir.mkdir(parents=True, exist_ok=True)
            step.artifacts["model_dir"] = str(model_dir)
            await self._log(f"[success]Created: {model_dir}[/success]")

        elif step.step_type == "install_packages":
            await self._log("[info]Installing huggingface_hub...[/info]")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", "--quiet",
                "huggingface_hub", "hf_transfer",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in proc.stdout:
                text = line.decode(errors="replace").rstrip()
                if text:
                    await self._log(f"  {text}")
            await proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("pip install huggingface_hub failed")
            step.artifacts["hub_installed"] = "1"

        elif step.step_type == "download_model":
            model_id = state.selected_model_id or ""
            model_dir = Path(
                state.steps[0].artifacts.get("model_dir", state.install_path or ".")
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            await self._log(f"[info]Downloading {model_id} → {model_dir}[/info]")
            await self._log("[muted]This may take a while for large models...[/muted]")

            # Select the best quantization if available in catalog
            try:
                entry = self._catalog.get_by_id(model_id)
                quant = entry.quantizations[0] if entry.quantizations else None
            except Exception:
                entry = None
                quant = None

            # Try GGUF single-file download first, then snapshot
            downloaded = await self._download_from_hub(model_id, model_dir, entry, quant)
            step.artifacts["downloaded_path"] = str(downloaded)
            await self._log(f"[success]Downloaded to: {downloaded}[/success]")

        elif step.step_type == "verify_install":
            downloaded = None
            for s in state.steps:
                if s.step_type == "download_model":
                    downloaded = s.artifacts.get("downloaded_path")
            if downloaded and Path(downloaded).exists():
                size_gb = sum(
                    f.stat().st_size for f in Path(downloaded).rglob("*") if f.is_file()
                ) / 1e9 if Path(downloaded).is_dir() else Path(downloaded).stat().st_size / 1e9
                await self._log(f"[success]Verified: {downloaded}  ({size_gb:.2f} GB)[/success]")
            else:
                await self._log("[warning]Could not verify — file path not found.[/warning]")

        elif step.step_type == "show_launch_info":
            downloaded = None
            for s in state.steps:
                if s.step_type == "download_model":
                    downloaded = s.artifacts.get("downloaded_path")
            model_id = state.selected_model_id or ""
            from model_manager.ui.console import console
            console.print("\n[header]── Launch Instructions ──[/header]")
            if downloaded and str(downloaded).endswith(".gguf"):
                console.print(f"  [info]llama.cpp:[/info]")
                console.print(f"    pip install llama-cpp-python")
                console.print(f"    python -c \"from llama_cpp import Llama; llm=Llama('{downloaded}'); print(llm('Hello')['choices'][0]['text'])\"")
                console.print(f"  [info]Ollama:[/info]")
                console.print(f"    ollama run {model_id.split('/')[-1].lower()}")
            else:
                console.print(f"  [info]Transformers:[/info]")
                console.print(f"    pip install transformers torch")
                console.print(f"    python -c \"from transformers import pipeline; p=pipeline('text-generation',model='{downloaded}'); print(p('Hello')[0]['generated_text'])\"")

    async def _download_from_hub(
        self,
        model_id: str,
        dest_dir: Path,
        entry=None,
        quant=None,
    ) -> Path:
        """Download model from HuggingFace Hub. Returns local path."""
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster downloads if hf_transfer present

        token = self._hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # For GGUF quantized models, download the single file
        if quant and quant.filename_pattern and entry:
            repo_id = quant.repo_url.split("huggingface.co/")[-1] if quant.repo_url else model_id
            # Resolve the actual filename from pattern
            filename = await self._resolve_gguf_filename(repo_id, quant.filename_pattern, token=token)
            if filename:
                await self._log(f"  Downloading file: {filename}")
                local = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest_dir),
                    token=token,
                )
                return Path(local)

        # Fallback: snapshot download (all files)
        await self._log(f"  Snapshot download of {model_id}...")
        local = await asyncio.to_thread(
            snapshot_download,
            repo_id=model_id,
            local_dir=str(dest_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            token=token,
        )
        return Path(local)

    async def _resolve_gguf_filename(self, repo_id: str, pattern: str, token: Optional[str] = None) -> Optional[str]:
        """Find a file in the HF repo matching the glob pattern."""
        try:
            from huggingface_hub import list_repo_files
            import fnmatch
            files = await asyncio.to_thread(list_repo_files, repo_id, token=token)
            for f in files:
                if fnmatch.fnmatch(f.lower(), pattern.lower().lstrip("*")):
                    return f
                if fnmatch.fnmatch(f, pattern):
                    return f
        except Exception:
            pass
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _handle_user_intervention(
        self,
        step,
        diagnosis,
        state: InstallationState,
    ) -> bool:
        """
        Show manual-action instructions when automated recovery fails.
        Returns True if the failed step should be retried.
        """
        import os
        from model_manager.ui.console import console

        console.print("\n[header]── Manual Action Required ──[/header]")
        if diagnosis.user_explanation:
            console.print(f"[warning]{diagnosis.user_explanation}[/warning]")

        root_cause = (diagnosis.root_cause or "").lower()
        is_auth = (
            diagnosis.error_category in ("auth_error", "access_denied")
            or "401" in root_cause
            or "token" in root_cause
            or "huggingface" in root_cause
            or "gated" in root_cause
        )

        if is_auth:
            model_id = state.selected_model_id or ""
            model_url = f"https://huggingface.co/{model_id}" if model_id else "https://huggingface.co"
            console.print("\n[header]Steps to fix:[/header]")
            console.print(f"  1. Open [info]{model_url}[/info] in your browser and accept the model license")
            console.print("  2. Go to [info]https://huggingface.co/settings/tokens[/info] and create a read-scope token")
            console.print("  3. Paste your token below — it will be used for this session only\n")

            raw_input = (await self._chat_input.get_input("HuggingFace token (or /cancel to abort): ")).strip()
            if raw_input.lower() == "/cancel" or not raw_input:
                return False

            # Extract bare token — user may paste surrounding text like "token: 'hf_xxx'"
            import re as _re
            m = _re.search(r"hf_[A-Za-z0-9]+", raw_input)
            token = m.group(0) if m else raw_input.strip("'\"\t ")

            if not token:
                await self._log("[warning]No token found in input — please paste just the token.[/warning]")
                return False

            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            self._hf_token = token
            await self._log(f"[success]Token accepted ({token[:8]}...) — retrying download...[/success]")
            return True

        # Generic: list required manual steps then ask to retry
        if diagnosis.requires_user_decision and diagnosis.decision_options:
            console.print("\n[header]Steps to fix:[/header]")
            for i, opt in enumerate(diagnosis.decision_options, 1):
                console.print(f"  {i}. {opt}")

        console.print()
        raw = (await self._chat_input.get_input("Press Enter to retry after completing the steps above (or /cancel to abort): ")).strip()
        if raw.lower() == "/cancel":
            return False
        return True

    async def _log(self, message: str) -> None:
        await bus.emit(LogLineEvent(level="info", message=message))

    async def _handle_slash(
        self, cmd: str, state: InstallationState, sm: StateMachine
    ) -> None:
        from model_manager.ui.console import console
        if cmd == "/help":
            from model_manager.ui.chat import SLASH_COMMANDS
            for c, desc in SLASH_COMMANDS.items():
                console.print(f"  [info]{c}[/info]  {desc}")
        elif cmd == "/status":
            console.print(f"  State: [info]{sm.current}[/info]")
            console.print(f"  Model: {state.selected_model_id or '—'}")
        elif cmd == "/sessions":
            for s in self._store.list_sessions():
                console.print(f"  {s['session_id'][:8]}  {s['current_state']}  {s['model']}")
        elif cmd == "/cancel":
            raise _StopLoop()
        elif cmd == "/exit":
            raise _StopLoop()

    def _extract_path_from_message(self, text: str) -> Optional[str]:
        """Extract a Windows or Unix path from a user message."""
        import re
        # Quoted Windows path: 'D:\...' or "D:\..."
        m = re.search(r"['\"]([A-Za-z]:\\[^'\"]+)['\"]", text)
        if m:
            return m.group(1).rstrip("\\/")
        # Quoted Unix path
        m = re.search(r"['\"](/[^'\"]+)['\"]", text)
        if m:
            return m.group(1)
        # Bare Windows path after keyword: "under D:\foo\bar" or "path D:\my models\..."
        # Allow spaces in path (stop at sentence-ending punctuation or end of string)
        m = re.search(r"(?:under|path|to|at|in)\s+([A-Za-z]:\\[^,;'\"]+?)(?:\s*$|\s+(?:please|now|ok|and)\b)", text, re.IGNORECASE)
        if m:
            return m.group(1).rstrip("\\/ \t")
        return None

    def _fuzzy_match_model_from_text(self, text: str):
        """
        Match a model from user text using family + size hints.
        E.g. 'gemma4 1b' → google/gemma-4-1b-it, '7b llama' → Llama 7B entry.
        Priority: exact size+family > partial match.
        """
        import re
        text_lower = text.lower()
        # Extract size hint: "1b", "4b", "7b", "12b", "27b" etc.
        size_m = re.search(r"(\d+\.?\d*)\s*b\b", text_lower)
        size_str = size_m.group(1) if size_m else None

        families = {
            "gemma":   ["gemma"],
            "llama":   ["llama"],
            "qwen":    ["qwen"],
            "mistral": ["mistral"],
        }
        matched_family = None
        for fam, keywords in families.items():
            if any(kw in text_lower for kw in keywords):
                matched_family = fam
                break

        if not matched_family and not size_str:
            return None  # not specific enough

        best = None
        for entry in self._catalog.all():
            id_lower = entry.model_id.lower()
            fam_lower = (entry.family or "").lower()
            # Family must match if user mentioned one
            if matched_family and matched_family not in fam_lower and matched_family not in id_lower:
                continue
            # Size must match if user mentioned one
            if size_str:
                param = entry.parameter_count_b
                # e.g. size_str="1" matches param=1.0; "0.5" matches 0.5
                try:
                    if abs(float(size_str) - param) > 0.1:
                        continue
                except ValueError:
                    continue
            # Passed all filters — this is a match
            if best is None:
                best = entry
            # Prefer the entry whose id is a closer substring match
            elif size_str and str(size_str) in id_lower:
                best = entry
        return best

    def _try_parse_install_signal(self, llm_response: str):
        """Parse [INSTALL: model_id] emitted by LLM when user confirms a model."""
        import re
        m = re.search(r'\[INSTALL:\s*([^\]]+)\]', llm_response)
        if not m:
            return None
        model_id = m.group(1).strip()
        try:
            return self._catalog.get_by_id(model_id)
        except Exception:
            # Try fuzzy match by display name substring
            needle = model_id.lower()
            for entry in self._catalog.all():
                if needle in entry.model_id.lower() or needle in entry.display_name.lower():
                    return entry
            return None

    def _try_parse_model_selection(self, text: str):
        """Return ModelEntry if user typed a number or exact model_id."""
        text = text.strip()
        if text.isdigit():
            if self._hardware:
                recs = ModelRecommender(self._catalog).recommend(self._hardware)
                idx = int(text) - 1
                if 0 <= idx < len(recs):
                    return recs[idx].model
        try:
            return self._catalog.get_by_id(text)
        except Exception:
            return None


class _StopLoop(Exception):
    """Sentinel to break out of the chat input loop."""
