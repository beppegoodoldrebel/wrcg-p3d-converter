# Read and write WRC Generations .P3D files (KT Engine "SCENE___").
#
# Two layers:
#   1. Chunk tree: parse() / serialize() keep every chunk, known or not, so an unchanged file is written back
#      byte for byte.
#   2. Geometry: Geometry(tree) splits the shared vertex/index buffer (GeHe/GHda) into its blocks (one per
#      stream of each part, plus the index list) and rebuild() lays them out again, recomputing the offsets in
#      SuGe, the size in GSiz and the GHda chunks. Blocks can be replaced with new data of any length.
#
# Layout learnt from the game files:
#   - chunk = 4-byte id + u32 size + data, padded to an even size; RIFF/LIST add a 4-byte kind
#   - most leaf chunks end with the marker CD AB
#   - GSiz: u16 1, u32 buffer size, u32 GHda chunk size (1 MB), u32 resource count, CD AB
#   - GHda: the buffer split in chunks of that size
#   - SuGe: u16 version, u32 vertex count, u32 stream count, (4-char name, u32 offset) per stream,
#           u32 triangle count, u32 index offset, then more fields kept as they are
#   - vertex streams start at a multiple of 16 bytes, index lists at a multiple of 4 (seen: right after the
#     previous stream); the gaps are zero bytes. Stream strides are not stored: they follow from the layout.

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

CONTAINERS = (b"RIFF", b"LIST")
STREAM_ALIGN = 16
INDEX_ALIGN = 4


@dataclass
class Chunk:
    id: bytes
    kind: Optional[bytes] = None          # RIFF/LIST only
    data: bytes = b""                     # leaf only
    children: List["Chunk"] = field(default_factory=list)
    pad: bytes = b""                      # the padding byte after an odd size, as found in the file

    def find(self, cid):
        """First direct child with this id (leaf) or kind (list)."""
        return next((c for c in self.children if c.id == cid or c.kind == cid), None)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def parse_range(b, start, end):
    out, pos = [], start
    while pos + 8 <= end:
        cid = b[pos:pos + 4]
        size = struct.unpack_from("<I", b, pos + 4)[0]
        body_end = pos + 8 + size
        if body_end > end:
            raise ValueError(f"chunk {cid!r} at {pos:#x} runs past its parent")
        pad = b[body_end:body_end + 1] if size & 1 else b""
        if cid in CONTAINERS:
            out.append(Chunk(cid, b[pos + 8:pos + 12], children=parse_range(b, pos + 12, body_end), pad=pad))
        else:
            out.append(Chunk(cid, data=b[pos + 8:body_end], pad=pad))
        pos = body_end + (size & 1)
    if pos != end:
        raise ValueError(f"{end - pos} unread bytes at {pos:#x}")
    return out


def parse(b):
    root = parse_range(b, 0, len(b))
    if len(root) != 1 or root[0].id != b"RIFF" or root[0].kind != b"PA__":
        raise ValueError("not a P3D file")
    return root[0]


def serialize(c):
    if c.id in CONTAINERS:
        body = c.kind + b"".join(serialize(x) for x in c.children)
    else:
        body = c.data
    out = c.id + struct.pack("<I", len(body)) + body
    if len(body) & 1:
        out += c.pad or b"\0"
    return out


def load(path):
    return parse(Path(path).read_bytes())


class Geometry:
    """Blocks of the shared buffer. Each block: dict(kind, part, name, data, ref) where kind is 'stream' or
    'index', part is the SuGe chunk and ref tells rebuild() where to write the new offset."""

    def __init__(self, root):
        self.root = root
        self.gehe = root.find(b"GeHe")
        self.gsiz = self.gehe.find(b"GSiz")
        ver, size, self.ghda_size, self.resources = struct.unpack_from("<HIII", self.gsiz.data, 0)
        ghda = [c for c in self.gehe.children if c.id == b"GHda"]
        if any(len(c.data) > self.ghda_size for c in ghda):
            raise ValueError("GHda chunk bigger than the size in GSiz")
        buf = b"".join(c.data for c in ghda)
        if len(buf) != size:
            raise ValueError("GSiz does not match the GHda chunks")
        self.parts = [c for c in root.walk() if c.id == b"SuGe"]
        refs = []   # (offset, part, slot): slot = stream index, or -1 for the indices
        for p in self.parts:
            nv, ns = struct.unpack_from("<II", p.data, 2)
            for i in range(ns):
                refs.append((struct.unpack_from("<I", p.data, 10 + i * 8 + 4)[0], p, i))
            refs.append((struct.unpack_from("<I", p.data, 10 + ns * 8 + 4)[0], p, -1))
        starts = sorted({r[0] for r in refs})
        # a block runs to the next block start; trailing zero padding is not part of it
        self.blocks, by_offset = [], {}
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else len(buf)
            data = buf[s:e]
            want = self._length(next(r for r in refs if r[0] == s), len(data))
            if any(data[want:]):
                raise ValueError(f"non-zero bytes in the padding at {s + want:#x}")
            first = next(r for r in refs if r[0] == s)
            block = dict(offset=s, data=data[:want], users=[], align=INDEX_ALIGN if first[2] == -1 else STREAM_ALIGN)
            by_offset[s] = block
            self.blocks.append(block)
        for off, p, slot in refs:
            by_offset[off]["users"].append((p, slot))
        last = self.blocks[-1]
        self.pad_end = len(buf) > last["offset"] + len(last["data"])   # whether the buffer ends padded
        self.original = buf

    @staticmethod
    def suge_fields(p):
        nv, ns = struct.unpack_from("<II", p.data, 2)
        streams = [(p.data[10 + i * 8:14 + i * 8].decode("latin-1"), struct.unpack_from("<I", p.data, 14 + i * 8)[0])
                   for i in range(ns)]
        ntri, ioff = struct.unpack_from("<II", p.data, 10 + ns * 8)
        return nv, streams, ntri, ioff

    def _length(self, ref, span):
        off, p, slot = ref
        nv, streams, ntri, ioff = self.suge_fields(p)
        if slot == -1:
            n = ntri * 3 * (2 if nv <= 0x10000 else 4)
            if not span - STREAM_ALIGN < n <= span:
                raise ValueError(f"index block of {n} bytes in a span of {span}")
            return n
        # the stride is not stored: the block fills its span up to the next aligned start. With very few
        # vertices several strides fit the span, so the stride known for the stream name is tried first
        # (a 3-vertex part, e.g. one empty triangle, could otherwise be cut short)
        known = {"vpos": 12, "vnor": 8, "vcol": 4, "vtan": 4, "vski": 8}.get(streams[slot][0])
        if known and span - STREAM_ALIGN < nv * known <= span:
            return nv * known
        for stride in range(4, 64, 4):
            if span - STREAM_ALIGN < nv * stride <= span:
                return nv * stride
        raise ValueError(f"no stride fits stream {streams[slot][0]} ({nv} vertices, span {span})")

    def add_part(self, suge, streams, index_data):
        """New blocks for a new part: streams = [(name, bytes)] in the order of its SuGe."""
        for slot, (name, data) in enumerate(streams):
            self.blocks.append(dict(offset=-1, data=data, users=[(suge, slot)], align=STREAM_ALIGN))
        self.blocks.append(dict(offset=-1, data=index_data, users=[(suge, -1)], align=INDEX_ALIGN))
        self.parts.append(suge)

    def drop_part(self, suge):
        """Forgets the blocks of a removed part."""
        for blk in self.blocks:
            blk["users"] = [u for u in blk["users"] if u[0] is not suge]
        self.blocks = [b for b in self.blocks if b["users"]]
        self.parts = [p for p in self.parts if p is not suge]

    def tree_order(self):
        """Blocks in the order of the game files: parts in MeLi order, each part's streams in SuGe order and
        then its index list. A part laid out elsewhere (e.g. a new part at the end of the buffer) loads but is
        not drawn by the game."""
        slot_of = {}
        for blk in self.blocks:
            for suge, slot in blk["users"]:
                slot_of.setdefault(id(suge), []).append((slot if slot >= 0 else 1 << 30, blk))
        out, seen = [], set()
        for c in self.root.walk():
            if c.id == b"SuGe":
                for slot, blk in sorted(slot_of.get(id(c), []), key=lambda x: x[0]):
                    if id(blk) not in seen:
                        seen.add(id(blk))
                        out.append(blk)
        out += [b for b in self.blocks if id(b) not in seen]
        return out

    def rebuild(self, layout="tree"):
        """Lays the blocks out again and updates SuGe, GSiz and GHda. layout "tree": the order of the parts in
        the file (as the game needs); "current": the order of self.blocks (used by the checks)."""
        if layout == "tree":
            self.blocks = self.tree_order()
        out = bytearray()
        new_off = {}
        for blk in self.blocks:
            out += b"\0" * (-len(out) % blk["align"])
            new_off[id(blk)] = len(out)
            out += blk["data"]
        if self.pad_end:
            out += b"\0" * (-len(out) % STREAM_ALIGN)
        for blk in self.blocks:
            for p, slot in blk["users"]:
                d = bytearray(p.data)
                ns = struct.unpack_from("<I", d, 6)[0]
                at = 14 + slot * 8 if slot >= 0 else 10 + ns * 8 + 4
                struct.pack_into("<I", d, at, new_off[id(blk)])
                p.data = bytes(d)
            blk["offset"] = new_off[id(blk)]
        d = bytearray(self.gsiz.data)
        struct.pack_into("<I", d, 2, len(out))
        self.gsiz.data = bytes(d)
        first = next(i for i, c in enumerate(self.gehe.children) if c.id == b"GHda")
        others = [c for c in self.gehe.children if c.id != b"GHda"]
        new = [Chunk(b"GHda", data=bytes(out[i:i + self.ghda_size])) for i in range(0, len(out), self.ghda_size)]
        self.gehe.children = others[:first] + new + others[first:]
        return bytes(out)
