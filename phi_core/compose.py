"""Compose framework: filter-graph execution over triple streams.

ffmpeg analogy (deliberate): models register like codecs; a pipeline
spec wires them like -filter_complex; the scheduler runs nodes when
inputs are ready; toggles enable/disable nodes at runtime (like
sendcmd/enable expressions). Arbitrary N models: registry + spec,
no code changes.

Core discipline (from LIBRARY_NOTES evidence):
- Stream descriptors travel WITH payloads (geometry/scale/layout in one
  header struct) — stale-metadata class eliminated structurally.
- Scales unify at seams where possible (same m = pointer pass, zero
  re-quantization); otherwise explicit rescale ops (auditable).
- Handoffs are triples (never decode/re-encode between models).
- Schedules are explicit data (schedules-as-semantics), versioned here.
"""
import numpy as np


class Stream:
    """Named triple payload + metadata header (travels together)."""

    def __init__(self, name, s, e, z, H, W, C, m=None, layout="HWC"):
        self.name = name
        assert s.shape == e.shape == z.shape
        self.s, self.e, self.z = s, e, z
        self.H, self.W, self.C = int(H), int(W), int(C)
        self.m = m
        self.layout = layout

    @property
    def n(self):
        return self.H * self.W * self.C

    def check(self, want=None):
        """Validate geometry (fail loud on stale metadata)."""
        assert (self.s.size, self.e.size, self.z.size) == (self.n,) * 3, \
            (self.name, self.s.shape, (self.H, self.W, self.C))
        if want:
            for k, v in want.items():
                assert getattr(self, k) == v, (self.name, k, getattr(self, k), v)
        return self


class Node:
    """One AI (or stage): named inputs -> named outputs, toggleable."""

    def __init__(self, name, fn, inputs, outputs):
        self.name = name
        self.fn = fn
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.on = True

    def run(self, feeds):
        assert self.on, f"node {self.name} is off (scheduler must skip)"
        for k in self.inputs:
            assert k in feeds, f"{self.name} missing input {k}"
        return self.fn(feeds)


class Registry:
    def __init__(self):
        self.nodes = {}

    def register(self, node):
        assert node.name not in self.nodes
        self.nodes[node.name] = node
        return node

    def toggle(self, name, on):
        self.nodes[name].on = bool(on)


class Scheduler:
    """Topological static order per frame; toggles rewire at runtime.

    Spec: "a=on:b=off" pairs (ffmpeg -filter_complex spirit). OFF nodes
    are SKIPPED: downstream nodes receive the named bypass (each OFF
    node declares `bypass`: output name -> input name to forward).
    """

    def __init__(self, registry, order, bypass=None):
        self.reg = registry
        self.order = list(order)
        self.bypass = bypass or {}

    def spec(self, s):
        """Apply "a=on:b=off" spec string. Unknown names fail loud."""
        for part in s.split(":"):
            name, state = part.split("=")
            assert name in self.reg.nodes, f"unknown node {name}"
            assert state in ("on", "off"), state
            self.reg.toggle(name, state == "on")
        return self

    def run(self, feeds):
        feeds = dict(feeds)
        for name in self.order:
            node = self.reg.nodes[name]
            if not node.on:
                for o, i in self.bypass.get(name, {}).items():
                    assert i in feeds, f"bypass {name}: missing {i}"
                    feeds[o] = feeds[i]
                continue
            out = node.run(feeds)
            feeds.update(out)
        return feeds
