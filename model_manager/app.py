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


def _is_connection_error(exc: Exception) -> bool:
    """Return True for network-level errors (not auth, rate-limit, or parse errors)."""
    type_name = type(exc).__name__.lower()
    msg       = str(exc).lower()
    return (
        "connect" in type_name
        or "network" in type_name
        or "connection" in msg
        or "timed out" in msg
        or "timeout" in msg
        or "ssl" in msg
        or "name resolution" in msg
        or "unreachable" in msg
        or "eof" in msg            # httpx EOFError on abrupt close
    )


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

        # Verify catalog repos exist on HuggingFace (fast concurrent HEAD checks)
        import os
        await self._log("[muted]Verifying catalog model availability...[/muted]")
        token = self._hf_token or os.environ.get("HF_TOKEN")
        validation = await self._catalog.validate_repos(token=token, timeout=5.0)
        removed = [mid for mid, ok in validation.items() if not ok]
        if removed:
            await self._log(
                f"[warning]Removed {len(removed)} unavailable model(s) from catalog: "
                + ", ".join(removed) + "[/warning]"
            )

        # Build system prompt with hardware + catalog injected
        system_prompt = self._build_recommendation_system_prompt()

        # Try to start the LLM conversation with full error recovery.
        if not await self._start_conversation_with_recovery(system_prompt):
            await self._fallback_to_simple_selection(state, sm)
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

            # Search HuggingFace when user mentions a model that may be recently released,
            # then inject the live search results so the LLM has up-to-date information.
            llm_input = text
            search_query = self._extract_model_search_query(text)
            if search_query:
                await self._log(f"[muted]Searching HuggingFace for '{search_query}'...[/muted]")
                search_results = await self._search_hf_models(search_query)
                if search_results:
                    search_ctx = self._format_hf_search_results(search_results)
                    llm_input = (
                        f"[HuggingFace Search Results for '{search_query}']\n"
                        f"{search_ctx}\n\n"
                        f"[User Message]\n{text}"
                    )
                    await self._log(
                        f"[muted]Injected {len(search_results)} HF search result(s) for '{search_query}'.[/muted]"
                    )

            response = await self._conversation.stream_response(llm_input, system=system_prompt)

            # Parse and validate any [RECOMMEND: ...] tags the LLM emitted
            added = await self._parse_and_validate_recommendations(response)
            if added:
                await self._log(
                    f"[muted]Validated and registered {len(added)} model(s): "
                    + ", ".join(added) + "[/muted]"
                )

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

    async def _parse_and_validate_recommendations(self, response: str) -> list[str]:
        """
        Parse [RECOMMEND: ...] tags from an LLM response.
        HEAD-validates each repo on HuggingFace and adds valid ones to the catalog.
        Returns list of model_ids that were successfully added.
        """
        import os
        import re
        import httpx
        from model_manager.catalog.models import ModelEntry, QuantizationOption

        token = self._hf_token or os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        added: list[str] = []

        matches = re.findall(r'\[RECOMMEND:\s*([^\]]+)\]', response)
        if not matches:
            return added

        async with httpx.AsyncClient(timeout=5.0) as client:
            for raw in matches:
                parts: dict[str, str] = {}
                for segment in raw.split("|"):
                    if "=" in segment:
                        k, _, v = segment.partition("=")
                        parts[k.strip()] = v.strip()

                repo_id = parts.get("repo_id", "").strip("/")
                if not repo_id:
                    continue

                # Validate the repo exists on HuggingFace
                api_url = f"https://huggingface.co/api/models/{repo_id}"
                try:
                    r = await client.head(api_url, headers=headers)
                    if r.status_code >= 400:
                        await self._log(
                            f"[muted]Skipping {repo_id} — not found on HuggingFace (HTTP {r.status_code})[/muted]"
                        )
                        continue
                except Exception:
                    pass  # network error — keep the recommendation

                def _to_float(val: str, default: float) -> float:
                    """Parse a numeric string that may carry a unit suffix (e.g. '16GB', '8.0 GB')."""
                    import re as _re
                    m = _re.search(r"[\d.]+", val)
                    return float(m.group()) if m else default

                quant_type  = parts.get("quant", "Q4_K_M")
                min_vram    = _to_float(parts.get("min_vram", "0.0"), 0.0)
                min_ram     = _to_float(parts.get("min_ram",  "4.0"), 4.0)
                file_size   = _to_float(parts.get("file_size","0.0"), 0.0)
                params      = _to_float(parts.get("params",   "0"),   0.0)
                display     = parts.get("name", repo_id.split("/")[-1])
                note        = parts.get("note", "")
                vendor      = repo_id.split("/")[0] if "/" in repo_id else "unknown"

                quant = QuantizationOption(
                    quant_type=quant_type,
                    file_size_gb=file_size,
                    min_vram_gb=min_vram,
                    quality_score=0.85,
                    repo_url=f"https://huggingface.co/{repo_id}",
                    filename_pattern=f"*{quant_type.lower()}*",
                )
                entry = ModelEntry(
                    model_id=repo_id,
                    display_name=display,
                    family=vendor.lower(),
                    parameter_count_b=params,
                    modality=["text"],
                    capabilities=["chat"],
                    license="unknown",
                    min_ram_gb=min_ram,
                    min_vram_gb=min_vram,
                    min_disk_gb=max(file_size * 1.1, 1.0),
                    supported_backends=["cuda", "cpu"] if min_vram > 0 else ["cpu"],
                    quantizations=[quant],
                    hf_repo_id=repo_id,
                    description=note,
                )
                self._catalog.add_entry(entry)
                added.append(repo_id)

        return added

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
        # Catalog is kept as a supplementary reference for verified models;
        # primary recommendations come from the LLM's own knowledge.
        catalog_section = self._format_catalog_for_llm()
        supplement = (
            f"\n\n## Supplementary Verified Models (catalog fallback)\n"
            f"The following models are pre-verified and can also be recommended.\n"
            f"You may recommend models OUTSIDE this list using [RECOMMEND:] tags.\n"
            f"{catalog_section}"
        ) if catalog_section.strip() else ""
        return f"{base}\n\n## User's Hardware\n{hw_section}{supplement}"

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

    async def _fallback_to_simple_selection(
        self, state: InstallationState, sm: StateMachine
    ) -> None:
        recs = ModelRecommender(self._catalog).recommend(self._hardware) if self._hardware else []
        rec_text = "\n".join(f"  {i+1}. {r.model.display_name}" for i, r in enumerate(recs))
        await self._simple_model_selection(state, sm, rec_text)

    # ── LLM startup with layered recovery ────────────────────────────────────

    async def _try_start_conversation(self, system_prompt: str) -> Optional[Exception]:
        """One attempt to create ConversationManager and warm it up. Returns None on success."""
        try:
            self._conversation = ConversationManager()
            await self._conversation.stream_response("Hi", system=system_prompt)
            return None
        except Exception as e:
            return e

    async def _start_conversation_with_recovery(self, system_prompt: str) -> bool:
        """
        Try to start the LLM conversation, with layered recovery:
          1. No API key      → prompt user to pick provider + enter key, then retry
          2. Connection error → offer proxy setup / retry / switch provider
          3. Other errors    → log and return False (caller falls back to simple selection)
        Returns True when self._conversation is ready to use.
        """
        exc = await self._try_start_conversation(system_prompt)
        if exc is None:
            return True

        # ── Layer 1: missing API key ──────────────────────────────────────────
        if isinstance(exc, APIKeyMissingError):
            provided = await self._prompt_api_key_setup()
            if not provided:
                await self._log("[warning]No API key provided — using simple model selection.[/warning]")
                return False
            exc = await self._try_start_conversation(system_prompt)
            if exc is None:
                return True

        # ── Layer 2: network / connection error ───────────────────────────────
        if exc is not None and _is_connection_error(exc):
            return await self._handle_connection_error(exc, system_prompt)

        # ── Layer 3: anything else (wrong key format, SDK version, etc.) ──────
        if exc is not None:
            await self._log(f"[warning]LLM unavailable ({exc}) — using simple model selection.[/warning]")
            return False

        return True

    async def _handle_connection_error(self, exc: Exception, system_prompt: str) -> bool:
        """
        Called when the LLM API is unreachable.
        Offers retry, proxy configuration, or provider switch.
        Returns True when self._conversation is ready.
        """
        import os
        from model_manager.ui.console import console

        console.print("\n[header]── Network Connection Failed ──[/header]")
        console.print(f"  [warning]{exc}[/warning]\n")
        console.print("  Cannot reach the LLM API. Common causes:")
        console.print("  • No internet connection")
        console.print("  • Firewall or VPN blocking the API endpoint")
        console.print("  • HTTP/HTTPS proxy required in your network\n")
        console.print("  1. Retry                    (check your connection first)")
        console.print("  2. Set HTTP proxy            (e.g. http://127.0.0.1:7890)")
        console.print("  3. Switch LLM provider      (try DeepSeek, OpenAI, etc.)")
        console.print("  4. Continue without AI      (simple model list)\n")

        raw = (await self._chat_input.get_input("Choice [1-4]: ")).strip()

        if raw == "1":
            exc2 = await self._try_start_conversation(system_prompt)
            if exc2 is None:
                await self._log("[success]Connected.[/success]")
                return True
            await self._log(f"[warning]Still unreachable: {exc2}[/warning]")
            return False

        elif raw == "2":
            proxy = (await self._chat_input.get_input(
                "  Proxy URL (e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:1080): "
            )).strip()
            if not proxy:
                return False
            os.environ["HTTPS_PROXY"] = proxy
            os.environ["HTTP_PROXY"]  = proxy
            await self._log(f"[info]Proxy set: {proxy} — retrying...[/info]")
            exc2 = await self._try_start_conversation(system_prompt)
            if exc2 is None:
                await self._log("[success]Connected via proxy.[/success]")
                return True
            await self._log(f"[warning]Still unreachable with proxy: {exc2}[/warning]")
            return False

        elif raw == "3":
            provided = await self._prompt_api_key_setup()
            if not provided:
                return False
            exc2 = await self._try_start_conversation(system_prompt)
            if exc2 is None:
                await self._log("[success]Connected.[/success]")
                return True
            if _is_connection_error(exc2):
                await self._log(f"[warning]Still unreachable: {exc2}[/warning]")
            else:
                await self._log(f"[warning]LLM unavailable: {exc2}[/warning]")
            return False

        # option 4 or invalid input
        return False

    async def _prompt_api_key_setup(self) -> bool:
        """
        Interactively ask the user to pick an LLM provider and enter an API key.
        Sets the matching environment variable so the next ConversationManager() call picks it up.
        Returns True when a key was successfully entered, False if the user skipped.
        """
        import os
        from model_manager.agent.factory import PROVIDER_ENV_VARS, LLMProvider
        from model_manager.ui.console import console

        PROVIDERS = [
            (LLMProvider.CLAUDE,   "Claude (Anthropic)",   "https://console.anthropic.com/settings/keys"),
            (LLMProvider.DEEPSEEK, "DeepSeek",             "https://platform.deepseek.com/api_keys"),
            (LLMProvider.QWEN,     "Qwen (Alibaba)",       "https://dashscope.console.aliyun.com/apiKey"),
            (LLMProvider.OPENAI,   "OpenAI",               "https://platform.openai.com/api-keys"),
            (LLMProvider.GEMINI,   "Gemini (Google)",      "https://aistudio.google.com/app/apikey"),
            (LLMProvider.MINIMAX,  "MiniMax",              "https://api.minimax.chat/"),
        ]

        console.print("\n[header]── No LLM API Key Detected ──[/header]")
        console.print("An LLM API key is needed for AI-powered model recommendations.\n")
        for i, (provider, name, _) in enumerate(PROVIDERS, 1):
            env_var = PROVIDER_ENV_VARS[provider]
            console.print(f"  {i}. {name:<24} [muted]({env_var})[/muted]")
        console.print(f"  {len(PROVIDERS) + 1}. Skip  [muted](use simple list selection instead)[/muted]")
        console.print()

        raw = (await self._chat_input.get_input(f"Select provider [1-{len(PROVIDERS) + 1}]: ")).strip()
        try:
            choice = int(raw)
        except ValueError:
            return False

        if choice == len(PROVIDERS) + 1:
            return False
        if not (1 <= choice <= len(PROVIDERS)):
            await self._log("[warning]Invalid choice — skipping.[/warning]")
            return False

        provider, name, key_url = PROVIDERS[choice - 1]
        env_var = PROVIDER_ENV_VARS[provider]

        console.print(f"\n  Get your {name} API key at: [info]{key_url}[/info]")
        key = (await self._chat_input.get_input(f"  Paste {name} API key: ")).strip()
        if not key:
            await self._log("[warning]Empty key — skipping.[/warning]")
            return False

        os.environ[env_var] = key
        await self._log(f"[success]{name} API key set ({key[:8]}...) — continuing.[/success]")
        return True

    # ── HuggingFace model search ───────────────────────────────────────────────

    def _extract_model_search_query(self, text: str) -> Optional[str]:
        """
        Detect model family + version mentions in user text that may refer to a
        recently released model (e.g. 'Gemma 4', 'Llama 4', 'Qwen3').
        Returns a short search query string, or None if nothing specific was found.
        """
        import re
        families = [
            "gemma", "llama", "qwen", "mistral", "deepseek", "phi",
            "falcon", "mixtral", "yi", "internlm", "baichuan", "glm",
            "chatglm", "nemotron", "minitron",
        ]
        text_lower = text.lower()
        for family in families:
            m = re.search(rf"\b{family}\s*(\d+(?:\.\d+)?)\b", text_lower)
            if m:
                version = m.group(1)
                return f"{family} {version}"
        return None

    async def _search_hf_models(self, query: str, limit: int = 8) -> list[dict]:
        """Search HuggingFace for GGUF models matching *query*. Returns raw API dicts."""
        import httpx
        import os
        token = self._hf_token or os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://huggingface.co/api/models",
                    params={
                        "search": query,
                        "filter": "gguf",
                        "sort": "downloads",
                        "direction": "-1",
                        "limit": limit,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return []

    def _format_hf_search_results(self, results: list[dict]) -> str:
        lines = []
        for r in results:
            repo_id   = r.get("id") or r.get("modelId", "")
            downloads = r.get("downloads", 0)
            likes     = r.get("likes", 0)
            tags      = [t for t in r.get("tags", []) if not t.startswith("license:")][:5]
            tag_str   = ", ".join(tags) if tags else ""
            lines.append(
                f"- {repo_id}  (downloads={downloads}, likes={likes}"
                + (f", tags: {tag_str}" if tag_str else "") + ")"
            )
        return "\n".join(lines) if lines else "No results found."

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
        backend, reason = await selector.select(
            self._hardware, user_preference=pref
        )
        state.backend = backend.name
        self._store.save(state)
        await self._log(f"Selected backend: [info]{backend.name}[/info]  ({reason})")
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
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # Handle permission failure directly — do NOT let it reach the LLM
                # recovery loop, which cannot update state.install_path and will loop forever.
                new_dir = await self._fix_mkdir_permission(model_dir, state)
                if new_dir is None:
                    raise PermissionError(
                        f"Cannot create {model_dir} — user aborted. "
                        f"Re-run with a writable path: mm --path ~/models"
                    )
                model_dir = new_dir
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

            # Ensure the HuggingFace cache dir is writable BEFORE the download starts.
            # If the default ~/.cache/huggingface was previously created by root (a common
            # side-effect of earlier sudo operations), the xet downloader will fail with
            # errno 13. We redirect HF_HOME to a path inside the model dir in that case.
            await self._ensure_hf_cache_writable(model_dir)

            await self._log(f"[info]Downloading {model_id} → {model_dir}[/info]")
            await self._log(
                "[muted]Download in progress — the HuggingFace progress bar shows live speed. "
                "A heartbeat line appears every 30 s so you can confirm the process is alive.[/muted]"
            )

            # Select the best quantization if available in catalog
            try:
                entry = self._catalog.get_by_id(model_id)
                quant = entry.quantizations[0] if entry.quantizations else None
            except Exception:
                entry = None
                quant = None

            # Run the download with a 30-second heartbeat so the user can see it's not stuck
            downloaded = await self._download_with_heartbeat(
                self._download_from_hub(model_id, model_dir, entry, quant)
            )
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
            console.print("\n[header]── How to Use Your Model ──[/header]")
            if downloaded and str(downloaded).endswith(".gguf"):
                model_path = downloaded
                short_name = model_id.split("/")[-1].lower().replace(".", "-")

                console.print("\n  [info]Option 1 — llama-cpp-python (Python, recommended)[/info]")
                console.print("    Install:")
                console.print("      pip install llama-cpp-python")
                console.print("    Interactive chat (run this script):")
                console.print(f'      from llama_cpp import Llama')
                console.print(f'      llm = Llama(model_path=r"{model_path}", n_ctx=4096, verbose=False)')
                console.print(f'      while True:')
                console.print(f'          user = input("You: ")')
                console.print(f'          if user.lower() in ("/exit", "/quit"): break')
                console.print(f'          out = llm.create_chat_completion(messages=[{{"role":"user","content":user}}])')
                console.print(f'          print("AI:", out["choices"][0]["message"]["content"])')

                console.print("\n  [info]Option 2 — Ollama (local server with API)[/info]")
                console.print("    Step 1 — Create a Modelfile (save as Modelfile, no extension):")
                console.print(f'      FROM {model_path}')
                console.print(f'      PARAMETER num_ctx 4096')
                console.print("    Step 2 — Import and run:")
                console.print(f"      ollama create {short_name} -f Modelfile")
                console.print(f"      ollama run {short_name}")

                console.print("\n  [info]Option 3 — llama-cpp-python OpenAI-compatible server[/info]")
                console.print("    Install:")
                console.print("      pip install llama-cpp-python[server]")
                console.print("    Start server:")
                console.print(f'      python -m llama_cpp.server --model "{model_path}" --n_ctx 4096')
                console.print("    Then call it like OpenAI API at http://localhost:8000")

            else:
                # Non-GGUF (transformers snapshot)
                console.print("\n  [info]Option 1 — Transformers (Python)[/info]")
                console.print("    Install:")
                console.print("      pip install transformers torch accelerate")
                console.print("    Interactive chat:")
                console.print(f'      from transformers import pipeline')
                console.print(f'      pipe = pipeline("text-generation", model=r"{downloaded}", device_map="auto")')
                console.print(f'      while True:')
                console.print(f'          user = input("You: ")')
                console.print(f'          if user.lower() in ("/exit", "/quit"): break')
                console.print(f'          out = pipe([{{"role":"user","content":user}}], max_new_tokens=512)')
                console.print(f'          print("AI:", out[0]["generated_text"][-1]["content"])')

                console.print("\n  [info]Option 2 — Ollama (import local model)[/info]")
                console.print(f"    ollama pull {model_id}")
                console.print(f"    ollama run {model_id.split('/')[-1].lower()}")

            # Docker option is applicable for GGUF models only (llama.cpp server image)
            if downloaded and str(downloaded).endswith(".gguf"):
                console.print("\n  [info]Option 4 — Docker (production / server deployment)[/info]")
                console.print("    Run the official llama.cpp server container:")
                console.print(f'      docker pull ghcr.io/ggerganov/llama.cpp:server')
                console.print(f'      docker run -p 8080:8080 \\')
                console.print(f'        -v "{downloaded}:/model.gguf:ro" \\')
                console.print(f'        ghcr.io/ggerganov/llama.cpp:server \\')
                console.print(f'        -m /model.gguf --host 0.0.0.0 --port 8080 -c 4096')
                console.print(f'      # OpenAI-compatible API will be available at http://localhost:8080')

            console.print("\n  [muted]When to use each option:[/muted]")
            console.print("  [muted]  pip / llama-cpp-python — personal use, development, quick start[/muted]")
            console.print("  [muted]  conda                  — NVIDIA/AMD GPU, need specific CUDA version[/muted]")
            console.print("  [muted]  Ollama                 — desktop app experience, easy model switching[/muted]")
            console.print("  [muted]  Docker                 — server/production, need isolation or API endpoint[/muted]")
            console.print("\n  [muted]Tip: use /exit or /quit to stop the Python chat loop.[/muted]")

    # ── Directory permission recovery ─────────────────────────────────────────

    async def _fix_mkdir_permission(
        self, target_dir: Path, state: InstallationState
    ) -> Optional[Path]:
        """
        Called when mkdir(target_dir) raises PermissionError.
        Presents the user with concrete options and returns the actual model_dir
        to use, or None if the user chooses to abort.
        Updating state.install_path here ensures subsequent steps use the right path.
        """
        import os
        import platform
        from model_manager.ui.console import console

        username = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
        home_alt = Path.home() / "models" / target_dir.name

        console.print(f"\n[header]── Cannot create directory ──[/header]")
        console.print(f"  [warning]Permission denied:[/warning] {target_dir}")
        console.print(f"  User '{username}' does not have write access to this path.\n")

        options: list[tuple[str, str]] = []
        if platform.system() != "Windows":
            options.append(("sudo",   f"Enter sudo password  (create {target_dir} and chown to {username})"))
        options.append(("home",   f"Use home directory   → {home_alt}"))
        options.append(("custom", "Enter a different path"))
        options.append(("abort",  "Abort installation"))

        for i, (_, label) in enumerate(options, 1):
            console.print(f"  {i}. {label}")
        console.print()

        raw = (await self._chat_input.get_input(f"Choice [1-{len(options)}]: ")).strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            return None
        if not (0 <= idx < len(options)):
            return None

        key = options[idx][0]

        if key == "sudo":
            return await self._mkdir_with_sudo(target_dir, username, state)
        elif key == "home":
            return await self._redirect_install_path(home_alt, state)
        elif key == "custom":
            raw_path = (await self._chat_input.get_input("  Install path: ")).strip()
            if not raw_path:
                return None
            custom_base = Path(raw_path).expanduser()
            custom_dir  = custom_base / target_dir.name
            return await self._redirect_install_path(custom_dir, state, base=custom_base)
        return None   # abort

    async def _mkdir_with_sudo(
        self, target_dir: Path, username: str, state: InstallationState
    ) -> Optional[Path]:
        """Create target_dir with sudo and chown it to the current user."""
        # Find the first non-existent ancestor — that's the one sudo must create.
        blocked = target_dir
        while blocked.parent != blocked and not blocked.parent.exists():
            blocked = blocked.parent

        password = (await self._chat_input.get_input("  sudo password: ")).strip()
        if not password:
            return None
        pw_bytes = (password + "\n").encode()

        async def _run_sudo(*args: str) -> tuple[int, str]:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-S", *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(input=pw_bytes), timeout=20)
            except asyncio.TimeoutError:
                proc.kill()
                return -1, "timed out"
            return proc.returncode, out.decode(errors="replace")

        rc, out = await _run_sudo("mkdir", "-p", str(target_dir))
        if rc != 0:
            await self._log(f"[warning]sudo mkdir failed: {out[:300]}[/warning]")
            await self._log("[info]Falling back to home directory...[/info]")
            home_alt = Path.home() / "models" / target_dir.name
            return await self._redirect_install_path(home_alt, state)

        # Transfer ownership so the process can write without elevation going forward
        rc2, out2 = await _run_sudo("chown", "-R", username, str(blocked))
        if rc2 != 0:
            await self._log(
                f"[warning]sudo chown failed: {out2[:200]} — "
                f"directory created but may need: sudo chown -R {username} {blocked}[/warning]"
            )

        await self._log(f"[success]Created via sudo: {target_dir}[/success]")
        return target_dir

    async def _redirect_install_path(
        self,
        new_dir: Path,
        state: InstallationState,
        base: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create new_dir, update state.install_path, and return new_dir.
        base, when given, is stored as state.install_path (the parent of new_dir);
        otherwise new_dir.parent is used.
        """
        try:
            new_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            await self._log(f"[error]Cannot create {new_dir}: {e}[/error]")
            return None
        state.install_path = str(base if base is not None else new_dir.parent)
        self._store.save(state)
        await self._log(f"[success]Redirected install path to: {new_dir}[/success]")
        return new_dir

    async def _ensure_hf_cache_writable(self, fallback_parent: Path) -> None:
        """
        Verify the HuggingFace cache directory is writable.
        If it was created by a previous root/sudo operation the current user cannot
        write there, which causes the xet downloader (huggingface_hub ≥ 1.x) to fail
        with errno 13. In that case we redirect HF_HOME to a sub-directory of the
        model install path, which is guaranteed to be writable.
        """
        import os
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        try:
            hf_home.mkdir(parents=True, exist_ok=True)
            probe = hf_home / ".mm_write_probe"
            probe.touch()
            probe.unlink()
        except (PermissionError, OSError):
            alt = fallback_parent / ".hf_cache"
            alt.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"]               = str(alt)
            os.environ["HUGGINGFACE_HUB_CACHE"]  = str(alt / "hub")
            await self._log(
                f"[info]HF cache redirected → {alt}  "
                f"(default path not writable by current user)[/info]"
            )

    async def _download_with_heartbeat(self, coro) -> Path:
        """
        Await *coro* (a download coroutine) while printing a status line every 30 s
        so the user can confirm the process is alive during long downloads.
        """
        async def _heartbeat() -> None:
            elapsed = 0
            while True:
                await asyncio.sleep(30)
                elapsed += 30
                m, s = divmod(elapsed, 60)
                await self._log(f"[muted]  ... download still running ({m}m {s:02d}s elapsed)[/muted]")

        hb = asyncio.create_task(_heartbeat())
        try:
            return await coro
        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass

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

        # For GGUF quantized models, download the single file from quant.repo_url
        if quant and quant.repo_url and entry:
            gguf_repo_id = quant.repo_url.split("huggingface.co/")[-1].strip("/")

            if quant.filename_pattern:
                filename = await self._resolve_gguf_filename(gguf_repo_id, quant.filename_pattern, token=token)
            else:
                filename = None

            if filename:
                await self._log(f"  Downloading file: {filename}")
                local = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=gguf_repo_id,
                    filename=filename,
                    local_dir=str(dest_dir),
                    token=token,
                )
                return Path(local)
            else:
                # Pattern match failed — snapshot the whole GGUF repo
                await self._log(f"  Snapshot download of GGUF repo {gguf_repo_id}...")
                local = await asyncio.to_thread(
                    snapshot_download,
                    repo_id=gguf_repo_id,
                    local_dir=str(dest_dir),
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
                    token=token,
                )
                return Path(local)

        # No quant info — snapshot download using model_id directly
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
            await self._log(f"[info]Token set ({token[:8]}...) — verifying access...[/info]")

            # Verify the token actually works before retrying
            model_id = state.selected_model_id or ""
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                user_info = await asyncio.to_thread(api.whoami)
                username = user_info.get("name", "?")
                await self._log(f"[success]Token valid (logged in as: {username})[/success]")

                # Check if the specific repo is accessible
                if model_id:
                    try:
                        entry = self._catalog.get_by_id(model_id)
                        quant = entry.quantizations[0] if entry.quantizations else None
                        repo_to_check = (
                            quant.repo_url.split("huggingface.co/")[-1].strip("/")
                            if quant and quant.repo_url else model_id
                        )
                    except Exception:
                        repo_to_check = model_id

                    try:
                        await asyncio.to_thread(api.repo_info, repo_to_check)
                        await self._log(f"[success]Repo access confirmed: {repo_to_check}[/success]")
                    except Exception as repo_err:
                        err_str = str(repo_err)
                        if "403" in err_str or "401" in err_str:
                            await self._log(
                                f"[warning]Token valid but repo access denied for {repo_to_check}.[/warning]\n"
                                f"[warning]You may need to accept the license at: "
                                f"https://huggingface.co/{repo_to_check}[/warning]"
                            )
                        elif "404" in err_str:
                            await self._log(
                                f"[warning]Repo not found: {repo_to_check} — the repo ID in the catalog may be wrong.[/warning]"
                            )
                        else:
                            await self._log(f"[warning]Repo check: {repo_err}[/warning]")
            except Exception as e:
                await self._log(f"[warning]Token may be invalid — whoami failed: {e}[/warning]")

            await self._log("[info]Retrying download...[/info]")
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
