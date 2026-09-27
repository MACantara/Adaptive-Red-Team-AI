"""Live TUI for the game — watch the adversary work, or defend it yourself.

Spectate mode renders ground truth from the per-node status codes every
log entry carries. Human mode renders only the redacted defender view —
alerts and metadata, never ownership. `--replay` scrubs a saved debrief.

Map codes (env.state_key per-node slot): 0 unseen, 1 discovered,
2 vulns-known, 3 owned, 4 rooted; +5 when isolated.
"""

import json
import queue
import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from artai.game import defenders as gdef
from artai.game.session import run_session

_STATUS = ["· unseen", "◦ found", "▽ vulns", "● owned", "▲ rooted"]
_TIER = {0: "edge", 1: "ws", 2: "srv", 3: "dc", 4: "jewel"}


def map_text(codes, env=None) -> Text:
    """Render per-node status codes as a color-tagged map."""
    t = Text()
    for i, code in enumerate(codes):
        isolated = code >= 5
        status = _STATUS[min(code - (5 if isolated else 0), 4)]
        tier = _TIER[env.net.nodes[i].tier] if env else f"n{i}"
        style = ("bright_red" if code % 5 >= 3 else
                 "yellow" if code % 5 == 2 else "green" if code else "grey50")
        t.append(f" n{i} {tier:5} ", style="bold")
        t.append(f"{status}{' [cut]' if isolated else ''}\n", style=style)
    return t


def entry_text(e) -> str:
    who = "ATK" if e["actor"] == "attacker" else "DEF"
    base = f"t{e['turn']:>2} {who} {e['action']}"
    if e.get("node") is not None:
        base += f" n{e['node']}"
    if e["actor"] == "attacker":
        tech = f" [{e['tech']}]" if e.get("tech") else ""
        flag = " ⚑DETECTED" if e.get("detected") else ""
        return f"{base}{tech} r={e.get('reward', 0):+.1f}{flag}"
    return f"{base} {'hit' if e.get('effected') else 'whiff'}"


class RedTeamApp(App):
    """Play or spectate a session live."""
    CSS = "#map {width: 32; border: solid green;} #log {border: solid cyan;}"

    def __init__(self, agent, env, defender_name, script_lines=None,
                 difficulty=None, spectate=True, eps=0.05, learn=False,
                 speed=0.15, seed=None):
        super().__init__()
        self.agent, self.env = agent, env
        self.spectate = spectate
        self.speed = speed
        self.seed = seed
        self._queue: queue.Queue = queue.Queue()
        self._last_view = None
        if defender_name == "human":
            self.defender_policy = self._human_turn
        else:
            self.defender_policy = gdef.resolve(
                defender_name, script_lines, difficulty=difficulty)
        self._run_args = dict(eps=eps, difficulty=difficulty, seed=seed,
                              train_lr=0.1 if learn else None)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Static(id="map")
            with Vertical():
                yield RichLog(id="log", wrap=True, markup=False)
                yield Input(placeholder="defender> pass | patch 4 | ...",
                            id="cmd", disabled=self.spectate)
        yield Footer()

    def on_mount(self):
        self.query_one("#map", Static).update("connecting…")
        self.run_worker(self._play, thread=True)

    def _play(self):
        report = run_session(self.agent, self.env, self.defender_policy,
                             on_event=self._on_event, **self._run_args)
        self.call_from_thread(self._finish, report)

    def _on_event(self, entry):
        time.sleep(self.speed)  # pace the worker; UI thread stays free
        self.call_from_thread(self._apply, entry)

    def _apply(self, entry):
        self.query_one("#log", RichLog).write(entry_text(entry))
        if self.spectate:
            self.query_one("#map", Static).update(
                map_text(entry["map"], self.env))

    def _human_turn(self, view):
        self._last_view = view
        self.call_from_thread(self._prompt_defender, view)
        return self._queue.get()

    def _prompt_defender(self, view):
        lines = Text()
        for nid, n in sorted(view["nodes"].items()):
            if n.alert > 0 or n.isolated or n.decoy:
                lines.append(
                    f" n{nid} {_TIER[n.tier]:5} alert={n.alert:.1f} "
                    f"tot={n.alert_total:.0f} vulns={n.vuln_count}"
                    f"{' [cut]' if n.isolated else ''}"
                    f"{' [decoy]' if n.decoy else ''}\n")
        self.query_one("#map", Static).update(lines or "quiet")
        self.query_one("#cmd", Input).disabled = False
        self.query_one("#cmd", Input).focus()

    def on_input_submitted(self, event: Input.Submitted):
        try:
            action = gdef.parse_line(event.value)
        except ValueError as e:
            self.query_one("#log", RichLog).write(f"!! {e}")
            return
        event.input.value = ""
        self._queue.put(action)

    def _finish(self, report):
        w = self.query_one("#log", RichLog)
        w.write(f"\n=== {report['result']} in {report['turns']} turns ===")
        w.write(f"kill chain: {' → '.join(k['tactic'] for k in report['kill_chain'])}")
        w.write(f"defender efficiency: {report['defender_efficiency']}")
        self._report = report


class ReplayApp(App):
    """Scrub a saved debrief: n/→ next, b/← back, q quit."""
    CSS = "#map {width: 32; border: solid green;} #log {border: solid cyan;}"
    BINDINGS = [("n", "step(1)", "next"), ("right", "step(1)", "next"),
                ("b", "step(-1)", "back"), ("left", "step(-1)", "back"),
                ("a", "autoplay", "auto"), ("q", "quit", "quit")]

    def __init__(self, report_path: Path):
        super().__init__()
        self.report = json.loads(Path(report_path).read_text())
        self.entries = self.report["log"]
        self.pos = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Static(id="map")
            yield RichLog(id="log", wrap=True)
        yield Footer()

    def on_mount(self):
        self.title = f"replay — {self.report['result']} in {self.report['turns']} turns"
        self._redraw()

    def action_step(self, delta):
        self.pos = max(0, min(len(self.entries), self.pos + delta))
        self._redraw()

    def action_autoplay(self):
        self.set_interval(0.4, self._tick)

    def _tick(self):
        if self.pos < len(self.entries):
            self.pos += 1
            self._redraw()

    def _redraw(self):
        log = self.query_one("#log", RichLog)
        log.clear()
        last_map = None
        for e in self.entries[:self.pos]:
            log.write(entry_text(e))
            if e.get("map"):
                last_map = e["map"]
        if last_map:
            self.query_one("#map", Static).update(map_text(last_map))
        kc = [k["tactic"] for k in self.report["kill_chain"]
              if k["turn"] <= (self.entries[self.pos - 1]["turn"]
                               if self.pos else -1)]
        self.sub_title = f"step {self.pos}/{len(self.entries)} — {' → '.join(kc)}"
