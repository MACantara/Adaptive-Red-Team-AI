from artai.env.network import generate


def test_seeded_generation_is_deterministic():
    a = generate(seed=42)
    b = generate(seed=42)
    assert a.snapshot() == b.snapshot()


def test_different_seeds_differ():
    assert generate(seed=1).snapshot() != generate(seed=2).snapshot()


def test_topology_shape():
    net = generate(seed=0)
    entry, cj = net.entry_point, net.crown_jewel
    assert net.nodes[entry].discovered
    assert net.nodes[cj].tier == 4
    # BFS reachability from entry to crown jewel
    seen, frontier = {entry}, [entry]
    while frontier:
        nxt = []
        for n in frontier:
            for m in net.links[n]:
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
    assert cj in seen


def test_guaranteed_dc_and_cj_access():
    net = generate(seed=7)
    assert any(net.crown_jewel in n.credentials for n in net.nodes.values())
    assert net.links[net.entry_point]  # entry must link somewhere
