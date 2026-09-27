"""Network graph model for the simulated enterprise.

Seeded, deterministic generation. No external graph lib needed at this size —
a dict of adjacency sets is plenty.
"""

from dataclasses import dataclass, field
import random


@dataclass
class Vuln:
    technique_id: str  # MITRE ATT&CK technique id from the catalog
    remote: bool = True  # remote-exploitable vs requires local user access


@dataclass
class Node:
    id: int
    name: str
    tier: int  # 0 entry / 1 workstation / 2 server / 3 dc / 4 crown jewel
    value: int = 1
    services: set[str] = field(default_factory=set)
    vulns: list[Vuln] = field(default_factory=list)
    local_vulns: list[Vuln] = field(default_factory=list)
    credentials: list[int] = field(default_factory=list)  # node ids creds unlock
    # runtime state
    discovered: bool = False
    owned: bool = False
    root: bool = False
    isolated: bool = False
    persistent: bool = False
    decoy: bool = False
    vulns_known: bool = False
    creds_known: bool = False
    alert: float = 0.0  # recent noise — decays each step
    alert_total: float = 0.0  # lifetime noise — never decays

    def snapshot(self) -> tuple:
        return (
            self.id, self.name, self.tier, self.value,
            tuple(sorted(self.services)),
            tuple((v.technique_id, v.remote) for v in self.vulns),
            tuple(v.technique_id for v in self.local_vulns),
            tuple(sorted(self.credentials)),
        )


@dataclass
class Network:
    nodes: dict[int, Node]
    links: dict[int, set[int]]
    entry_point: int
    crown_jewel: int

    def snapshot(self) -> tuple:
        return (
            self.entry_point,
            self.crown_jewel,
            tuple(self.nodes[i].snapshot() for i in sorted(self.nodes)),
            tuple(
                (i, tuple(sorted(self.links[i]))) for i in sorted(self.links)
            ),
        )

    def reachable(self, node_id: int) -> set[int]:
        out = set()
        for n in self.links.get(node_id, ()):
            if not self.nodes[n].isolated:
                out.add(n)
        return out


# exploit-action techniques only — a vuln entry whose catalog action isn't
# "exploit" is a dead slot its owner can never be taken through
_REMOTE_TECHNIQUES = ("T1190", "T1357", "T1210")
_LOCAL_TECHNIQUES = ("T1068", "T1078")
_SERVICES = ("http", "smb", "rdp", "ssh", "winrm", "mssql")
_TIER_NAMES = ("edge", "ws", "srv", "dc", "cj")
_TIER_VALUES = (2, 1, 2, 5, 10)


def generate(seed: int, n_workstations: int = 3, n_servers: int = 3) -> Network:
    """Deterministic layered topology: edge -> ws -> srv -> dc -> cj."""
    rng = random.Random(seed)
    nodes: dict[int, Node] = {}
    links: dict[int, set[int]] = {}

    def add(name: str, tier: int) -> int:
        nid = len(nodes)
        nodes[nid] = Node(
            id=nid,
            name=f"{name}-{nid}",
            tier=tier,
            value=_TIER_VALUES[tier],
            services=set(rng.sample(_SERVICES, rng.randint(1, 3))),
        )
        links[nid] = set()
        return nid

    def link(a: int, b: int) -> None:
        links[a].add(b)
        links[b].add(a)

    entry = add("edge", 0)
    ws = [add(_TIER_NAMES[1], 1) for _ in range(n_workstations)]
    srv = [add(_TIER_NAMES[2], 2) for _ in range(n_servers)]
    dc = add("dc", 3)
    cj = add("cj", 4)

    for w in ws:
        link(entry, w)
        for s in srv:
            if rng.random() < 0.5:
                link(w, s)
    for s in srv:
        link(s, dc)
    link(dc, cj)
    # redundant routes to the crown jewel — one burned node must not be
    # checkmate, or defender patch/reimage decides the game by seed alone
    for s in rng.sample(srv, min(2, len(srv))):
        link(s, cj)
    for i in range(len(srv)):
        for j in range(i + 1, len(srv)):
            if rng.random() < 0.4:
                link(srv[i], srv[j])
    # a few cross links among workstations for path diversity
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            if rng.random() < 0.3:
                link(ws[i], ws[j])

    for nid, node in nodes.items():
        p_remote = 0.9 if node.tier == 0 else 0.6
        for _ in range(rng.randint(0, 2)):
            if rng.random() < p_remote:
                node.vulns.append(
                    Vuln(technique_id=rng.choice(_REMOTE_TECHNIQUES))
                )
        if rng.random() < 0.7:
            node.local_vulns.append(Vuln(technique_id=rng.choice(_LOCAL_TECHNIQUES), remote=False))

        if node.tier in (2, 3):
            candidates = [n for n in nodes if nodes[n].tier > node.tier - 1 and n != nid]
            if candidates:
                node.credentials.append(rng.choice(candidates))
        # keep every node ownable — no unreachable pockets in the arena
        if node.tier > 0 and not node.vulns:
            node.vulns.append(Vuln(technique_id=rng.choice(_REMOTE_TECHNIQUES)))

    # guarantee a viable path of vulns for the heuristic baseline
    nodes[entry].vulns.append(Vuln(technique_id="T1190"))
    # at least two distinct credential holders unlock each deep node —
    # losing one to a patch/reimage leaves an alternate route
    for h in rng.sample(srv, min(2, len(srv))):
        if dc not in nodes[h].credentials:
            nodes[h].credentials.append(dc)
    for h in {dc, srv[-1]}:
        if cj not in nodes[h].credentials:
            nodes[h].credentials.append(cj)
    nodes[dc].local_vulns.append(Vuln(technique_id="T1068", remote=False))
    if not nodes[cj].local_vulns:
        nodes[cj].local_vulns.append(Vuln(technique_id="T1068", remote=False))

    nodes[entry].discovered = True
    return Network(nodes=nodes, links=links, entry_point=entry, crown_jewel=cj)
