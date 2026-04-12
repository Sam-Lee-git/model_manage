"""Simple sequential UI — no Rich Live, no terminal ownership conflicts."""

from __future__ import annotations

from typing import Optional

from rich.panel import Panel
from rich.table import Table

from model_manager.core.events import (
    BranchFailedEvent, BranchStartedEvent, BranchStepEvent, BranchSucceededEvent,
    ChatResponseChunkEvent, DiagnosisReadyEvent, DownloadProgressEvent,
    LogLineEvent, StateChangedEvent, StepCompletedEvent, StepFailedEvent,
    StepStartedEvent, bus,
)
from model_manager.hardware.profile import HardwareProfile
from model_manager.ui.console import console


class Dashboard:
    """
    Sequential terminal UI.
    Prints output as it arrives — no full-screen takeover, no stdin conflicts.
    """

    def __init__(self) -> None:
        self._hardware: Optional[HardwareProfile] = None
        self._stream_buffer = ""   # accumulate streaming chunks on one line
        self._register_handlers()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        pass   # nothing to start in sequential mode

    def stop(self) -> None:
        if self._stream_buffer:
            console.print()   # end any partial stream line
            self._stream_buffer = ""

    def set_hardware(self, hardware: HardwareProfile) -> None:
        self._hardware = hardware
        self._print_hardware(hardware)

    def add_chat_message(self, role: str, text: str) -> None:
        """Print a full chat message (non-streamed)."""
        if not text.strip():
            return
        if role == "agent":
            console.print(f"[bold green]Agent:[/bold green] {text}")
        else:
            console.print(f"[bold blue]You:[/bold blue] {text}")

    # ── Hardware display ──────────────────────────────────────────────────────

    def _print_hardware(self, h: HardwareProfile) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_row("[muted]CPU[/muted]",  h.cpu.brand)
        table.add_row("[muted]RAM[/muted]",  f"{h.ram_total_gb:.1f} GB total  {h.ram_available_gb:.1f} GB free")
        for gpu in h.gpus:
            table.add_row(
                f"[muted]{gpu.compute_backend.upper()}[/muted]",
                f"{gpu.name}  {gpu.vram_gb:.1f} GB VRAM",
            )
        for drive in h.drives[:4]:
            table.add_row(
                "[muted]Disk[/muted]",
                f"{drive.path}  {drive.free_gb:.0f}/{drive.total_gb:.0f} GB free",
            )
        console.print(Panel(table, title="[header]Hardware[/header]", expand=False))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _register_handlers(self) -> None:
        bus.subscribe(StateChangedEvent,      self._on_state_changed)
        bus.subscribe(StepStartedEvent,       self._on_step_started)
        bus.subscribe(StepCompletedEvent,     self._on_step_completed)
        bus.subscribe(StepFailedEvent,        self._on_step_failed)
        bus.subscribe(LogLineEvent,           self._on_log)
        bus.subscribe(ChatResponseChunkEvent, self._on_chat_chunk)
        bus.subscribe(DiagnosisReadyEvent,    self._on_diagnosis)
        bus.subscribe(BranchStartedEvent,     self._on_branch_started)
        bus.subscribe(BranchStepEvent,        self._on_branch_step)
        bus.subscribe(BranchSucceededEvent,   self._on_branch_succeeded)
        bus.subscribe(BranchFailedEvent,      self._on_branch_failed)
        bus.subscribe(DownloadProgressEvent,  self._on_download)

    async def _on_state_changed(self, e: StateChangedEvent) -> None:
        console.print(f"[muted]── {e.previous} → {e.current} ──[/muted]")

    async def _on_step_started(self, e: StepStartedEvent) -> None:
        console.print(f"[info]▶ [{e.step_index+1}/{e.total_steps}] {e.step_name}[/info]")

    async def _on_step_completed(self, e: StepCompletedEvent) -> None:
        console.print(f"[success]✓ {e.step_name}[/success]")

    async def _on_step_failed(self, e: StepFailedEvent) -> None:
        console.print(f"[error]✗ {e.step_name}: {e.error}[/error]")

    async def _on_log(self, e: LogLineEvent) -> None:
        console.print(e.message)

    async def _on_chat_chunk(self, e: ChatResponseChunkEvent) -> None:
        """Stream LLM output inline, flush on newlines. Hide internal [INSTALL:...] tokens."""
        import re
        _INSTALL_RE = re.compile(r'\[INSTALL:[^\]]*\]')

        if e.is_final:
            if self._stream_buffer:
                clean = _INSTALL_RE.sub("", self._stream_buffer).rstrip()
                if clean:
                    console.print(clean)
                self._stream_buffer = ""
            return
        self._stream_buffer += e.chunk
        # Flush complete lines immediately, skip lines that are just the install token
        while "\n" in self._stream_buffer:
            line, self._stream_buffer = self._stream_buffer.split("\n", 1)
            display = _INSTALL_RE.sub("", line).rstrip()
            if display:
                console.print(display)

    async def _on_diagnosis(self, e: DiagnosisReadyEvent) -> None:
        console.print(f"[bold yellow]Diagnosis:[/bold yellow] {e.root_cause}  [muted](confidence {e.confidence:.0%})[/muted]")

    async def _on_branch_started(self, e: BranchStartedEvent) -> None:
        console.print(f"[bold yellow]Recovery branch (depth={e.depth}): {e.fix_steps} steps[/bold yellow]")

    async def _on_branch_step(self, e: BranchStepEvent) -> None:
        console.print(f"  [muted]fix[{e.step_index+1}][/muted] {e.step_description}")

    async def _on_branch_succeeded(self, e: BranchSucceededEvent) -> None:
        console.print(f"[success]Recovery succeeded (depth={e.depth}), resuming...[/success]")

    async def _on_branch_failed(self, e: BranchFailedEvent) -> None:
        console.print(f"[error]Recovery failed (depth={e.depth}): {e.reason}[/error]")

    async def _on_download(self, e: DownloadProgressEvent) -> None:
        pct = e.downloaded_bytes / e.total_bytes * 100 if e.total_bytes else 0
        console.print(
            f"[info]↓ {e.filename}  {pct:.0f}%  {e.speed_mbps:.1f} MB/s[/info]",
            end="\r",
        )
