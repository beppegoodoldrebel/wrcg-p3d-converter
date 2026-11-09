# .PSI (scene instance) files: which material (.PMA) each part of each model node uses.
#
# RIFF "PA__": Head ("SCENINSTENDI"...), SIHe, then physics chunks (PhAC, PhAc, PhJC) kept as they are.
# SIHe: u16 3, u32 length + model path, u32 entry count, entries:
#   u16 2, u32 length + node name, u32 length + .PMA path, u32 part index in the node, u8 0, CD AB
# Entries follow the order of the parts.

import struct
from pathlib import Path

from p3d_lib import parse, serialize


def _str(d, p):
    n = struct.unpack_from("<I", d, p)[0]
    return d[p + 4:p + 4 + n].decode("latin-1"), p + 4 + n


def _pack(s):
    b = s.encode("latin-1")
    return struct.pack("<I", len(b)) + b


class Psi:
    def __init__(self, path):
        self.root = parse(Path(path).read_bytes())
        self.sihe = self.root.find(b"SIHe")
        d = self.sihe.data
        self.version = struct.unpack_from("<H", d, 0)[0]
        self.model, p = _str(d, 2)
        count = struct.unpack_from("<I", d, p)[0]
        p += 4
        self.entries = []      # [node, pma, part index, flag]
        for _ in range(count):
            if struct.unpack_from("<H", d, p)[0] != 2:
                raise ValueError("unexpected PSI entry")
            node, p = _str(d, p + 2)
            pma, p = _str(d, p)
            index, flag = struct.unpack_from("<IB", d, p)
            if d[p + 5:p + 7] != b"\xcd\xab":
                raise ValueError("PSI entry marker missing")
            self.entries.append([node, pma, index, flag])
            p += 7
        self.tail = d[p:]

    def _runs(self, node):
        """Entry index ranges of each node with this name: two nodes with the same name are two runs,
        each starting again from part index 0."""
        runs = []
        for i, e in enumerate(self.entries):
            if e[0] != node:
                continue
            if runs and runs[-1][1] == i and e[2] != 0:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        return runs

    def materials(self, node, occurrence=0):
        """[(.PMA path, flag)] of a node's parts, by part index (occurrence: which node with that name)."""
        runs = self._runs(node)
        if occurrence >= len(runs):
            return []
        a, b = runs[occurrence]
        return [(e[1], e[3]) for e in sorted(self.entries[a:b], key=lambda e: e[2])]

    def set_materials(self, node, occurrence, parts):
        """Replaces the node's entries with [(.PMA path, flag)], in part order, at the same place."""
        runs = self._runs(node)
        new = [[node, pma, k, flag] for k, (pma, flag) in enumerate(parts)]
        if occurrence < len(runs):
            a, b = runs[occurrence]
            self.entries[a:b] = new
        else:
            self.entries += new

    def pma_for(self, material):
        """First .PMA path whose file name is MATERIAL.PMA, or None."""
        for e in self.entries:
            if Path(e[1].replace("DATA:/", "")).stem.upper() == material.upper():
                return e[1]
        return None

    def to_bytes(self):
        out = bytearray(struct.pack("<H", self.version) + _pack(self.model) + struct.pack("<I", len(self.entries)))
        for node, pma, index, flag in self.entries:
            out += struct.pack("<H", 2) + _pack(node) + _pack(pma) + struct.pack("<IB", index, flag) + b"\xcd\xab"
        self.sihe.data = bytes(out + self.tail)
        return serialize(self.root)
