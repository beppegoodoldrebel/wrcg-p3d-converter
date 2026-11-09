# Wipers <-> Blender: the wiper meshes with their weights, bones and animation as one glTF, and back.
#
#   python p3d_wipers.py export MODEL.P3D WIPERS.PAN OUT.gltf
#   python p3d_wipers.py import MODEL.P3D WIPERS.PAN EDITED.glb OUT_FOLDER     (writes model, .PAN and .PSI there)
#
# In the glTF the bones are a real armature (Ref_Wipers and the animated D_Wiper_* bones), so Blender can edit
# the meshes, weights, bone positions and keyframes freely. REF_* meshes are bodywork for reference, never read.
#
# How the game draws a wiper vertex: v x Sum(weight x IB x B), v = stored vertex (node space), IB = the bone's
# matrix in the model's Skin chunk, B = the bone's world matrix, row vectors. WIPERS.PAN is played on an
# ADDITIVE layer (.PAR): only the motion from the first key counts, K(t) x K(0)^-1 applied to the bone's bind.
#
# Reading Blender's file back uses only MOTION from the first frame: Delta(t) = inverse(W(t0)) x W(t). Blender
# may add a fixed rotation to a bone's own axes on the way (it does on empties), and that cancels out here.
# So in the game each bone keeps its rotation axes and takes from Blender its position and its motion.
# The .PAN format has rotation keys only: a bone may rotate, its position relative to its parent is fixed
# (translation keys in Blender are reported and ignored).
#
# .PAN (RIFF PA__, Head + Data). Head: +4 u32 size, +8 f32 duration, +16 four u16 counts (static rotations,
# translations, scales, animated rotations). Data: the four lists of u16 bone indices (P3D RTbl), u32 cumulative
# key count per animated bone, f32 key times (4-aligned), then (4-aligned) static quaternions (x, y, z, w = of the
# row-vector matrix transposed), translations, scales, then 6-byte keys: smallest-three, bits 0-1 = omitted
# component, 15-bit fields, the field at bit 32 = first kept component, scale 1/sqrt(2).

import base64
import json
import math
import re
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obj_to_p3d import Nearest, apply, import_obj, invert, matmul, model_nodes, resources  # noqa: E402
from p3d_lib import Geometry, load, serialize  # noqa: E402
from p3d_to_obj import read_psi_materials  # noqa: E402

A = (1.0, 0, 0, 0, 0, 0, -1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0)     # game -> glTF axes, row vectors
Ai = invert(A)
IDENTITY = (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0)
REFERENCE_NODES = ("Glass_F", "Hood", "Body")
DRIFT = 0.002                     # metres a bone may drift from its parent in Blender before it is reported
to_gl = lambda p: (p[0], p[2], -p[1])
to_game = lambda p: (p[0], -p[2], p[1])
gl_matrix = lambda m: matmul(matmul(Ai, m), A)
game_matrix = lambda m: matmul(matmul(A, m), Ai)
rot3 = lambda m: [list(m[0:3]), list(m[4:7]), list(m[8:11])]


def with_translation(rows, t):
    return tuple(rows[0]) + (0.0,) + tuple(rows[1]) + (0.0,) + tuple(rows[2]) + (0.0,) + tuple(t) + (1.0,)


def quat_rows(q):
    """Row-vector 3x3 of a quaternion (x, y, z, w) as stored in a .PAN or a glTF."""
    x, y, z, w = q
    c = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return [[c[j][i] for j in range(3)] for i in range(3)]


def rows_quat(rows):
    """Inverse of quat_rows."""
    R = [[rows[c][r] for c in range(3)] for r in range(3)]
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        return [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, s / 4]
    i = max(range(3), key=lambda k: R[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + R[i][i] - R[j][j] - R[k][k]) * 2
    q = [0.0] * 4
    q[i], q[j], q[k], q[3] = s / 4, (R[j][i] + R[i][j]) / s, (R[k][i] + R[i][k]) / s, (R[k][j] - R[j][k]) / s
    return q


def mul3(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def orthonormal(rows):
    """Nearest proper rotation: Gram-Schmidt keeps a key from drifting."""
    x = rows[0]
    lx = math.sqrt(sum(v * v for v in x))
    x = [v / lx for v in x]
    y = rows[1]
    d = sum(a * b for a, b in zip(x, y))
    y = [b - d * a for a, b in zip(x, y)]
    ly = math.sqrt(sum(v * v for v in y))
    y = [v / ly for v in y]
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    return [x, y, z]


# ---------------------------------------------------------------- game files

class Pan:
    def __init__(self, path):
        self.root = load(path)
        self.head = next(c for c in self.root.children if c.id == b"Head")
        self.chunk = next(c for c in self.root.children if c.id == b"Data")
        h, d = self.head.data, self.chunk.data
        nR, nT, nS, nA = struct.unpack_from("<4H", h, 16)
        lists = struct.unpack_from(f"<{nR + nT + nS + nA}H", d, 0)
        self.rot, self.trans = list(lists[:nR]), list(lists[nR:nR + nT])
        self.scale, self.anim = list(lists[nR + nT:nR + nT + nS]), list(lists[nR + nT + nS:])
        o = 2 * len(lists)
        cum = struct.unpack_from(f"<{nA}I", d, o)
        total = cum[-1] if nA else 0
        t0 = (o + 4 * nA + 3) // 4 * 4
        times = struct.unpack_from(f"<{total}f", d, t0)
        s0 = (t0 + 4 * total + 3) // 4 * 4
        self.quats = [struct.unpack_from("<4f", d, s0 + 16 * i) for i in range(nR)]
        self.trans_values = [struct.unpack_from("<3f", d, s0 + 16 * nR + 12 * i) for i in range(nT)]
        self.scale_values = [struct.unpack_from("<3f", d, s0 + 16 * nR + 12 * nT + 12 * i) for i in range(nS)]
        k0 = s0 + 16 * nR + 12 * nT + 12 * nS
        self.tail = d[k0 + 6 * total:]
        self.size_extra = struct.unpack_from("<I", h, 4)[0] - len(d)
        self.keys, start = {}, 0
        for bone, end in zip(self.anim, cum):
            out = []
            for i in range(start, end):
                out.append((times[i], decode_key(d[k0 + 6 * i:k0 + 6 * i + 6])))
            self.keys[bone] = out
            start = end

    def save(self, path):
        d = bytearray()
        for lst in (self.rot, self.trans, self.scale, self.anim):
            d += struct.pack(f"<{len(lst)}H", *lst)
        cum, n = [], 0
        for b in self.anim:
            n += len(self.keys[b])
            cum.append(n)
        d += struct.pack(f"<{len(cum)}I", *cum)
        while len(d) % 4:
            d.append(0)
        for b in self.anim:
            d += b"".join(struct.pack("<f", t) for t, _ in self.keys[b])
        while len(d) % 4:
            d.append(0)
        for q in self.quats:
            d += struct.pack("<4f", *q)
        for lst in (self.trans_values, self.scale_values):
            for v in lst:
                d += struct.pack("<3f", *v)
        for b in self.anim:
            d += b"".join(encode_key(q) for _, q in self.keys[b])
        d += self.tail
        h = bytearray(self.head.data)
        struct.pack_into("<I", h, 4, len(d) + self.size_extra)
        duration = max(t for b in self.anim for t, _ in self.keys[b])
        struct.pack_into("<f", h, 8, duration)
        self.head.data, self.chunk.data = bytes(h), bytes(d)
        Path(path).write_bytes(serialize(self.root))


def decode_key(raw):
    u = struct.unpack("<3H", raw)
    v = u[0] | u[1] << 16 | u[2] << 32
    idx = v & 3
    f = [((v >> (2 + 15 * k)) & 0x7FFF) / 0x7FFF * 2 - 1 for k in range(3)]
    f = [f[2] / math.sqrt(2), f[1] / math.sqrt(2), f[0] / math.sqrt(2)]
    big = math.sqrt(max(0.0, 1 - sum(x * x for x in f)))
    return tuple(f[:idx] + [big] + f[idx:])


def encode_key(q):
    i = max(range(4), key=lambda k: abs(q[k]))
    if q[i] < 0:
        q = [-v for v in q]
    kept = [q[k] for k in range(4) if k != i]
    field = lambda v: max(0, min(0x7FFF, round((v * math.sqrt(2) + 1) / 2 * 0x7FFF)))
    v = i | field(kept[2]) << 2 | field(kept[1]) << 17 | field(kept[0]) << 32
    return struct.pack("<3H", v & 0xFFFF, (v >> 16) & 0xFFFF, (v >> 32) & 0xFFFF)


def bone_table(root):
    d = next(c for c in root.children if c.id == b"RTbl").data
    n = struct.unpack_from("<I", d, 2)[0]
    o, out = 6, []
    for _ in range(n):
        ln = struct.unpack_from("<I", d, o)[0]
        out.append(d[o + 4:o + 4 + ln].decode("latin-1"))
        o += 4 + ln
    return out


def read_skin(root):
    """{node: [(bone name, IB matrix)]} from the Skin chunk: the palette the vertices' bone indices refer to."""
    d = next(c for c in root.children if c.id == b"Skin").data
    count = struct.unpack_from("<I", d, 2)[0]
    o, out = 6, {}
    for _ in range(count):
        ln = struct.unpack_from("<I", d, o + 2)[0]
        name = d[o + 6:o + 6 + ln].decode("latin-1")
        o += 6 + ln + 8
        n = struct.unpack_from("<I", d, o)[0]
        o += 4
        pal = []
        for _ in range(n):
            l2 = struct.unpack_from("<I", d, o + 2)[0]
            pal.append((d[o + 6:o + 6 + l2].decode("latin-1"), struct.unpack_from("<16f", d, o + 6 + l2)))
            o += 6 + l2 + 64 + 2
        out[name] = pal
        if d[o:o + 2] == b"\xcd\xab":
            o += 2
    return out


def tree(root):
    """{node: (parent name or None, local matrix)} of the node tree (first node of each name)."""
    out = {}

    def walk(lst, parent):
        cur = None
        for c in lst:
            if c.id == b"NoHe":
                n = struct.unpack_from("<I", c.data, 2)[0]
                cur = c.data[6:6 + n].decode("latin-1")
                out.setdefault(cur, (parent, struct.unpack_from("<16f", c.data, 6 + n + 2)))
            elif c.children:
                walk(c.children, cur if cur is not None else parent)
    walk(root.children, None)
    return out


def part_data(geo_blocks, part):
    nv, st, nt, ioff = Geometry.suge_fields(part["suge"])
    st = dict(st)
    b = {k: geo_blocks[v]["data"] for k, v in st.items()}
    uvstride = (len(re.findall(r"UV\d_", part["format"])) or 1) * 4
    return dict(pos=[struct.unpack_from("<3f", b["vpos"], i * 12) for i in range(nv)],
                nor=[struct.unpack_from("<3e", b["vnor"], i * 8) for i in range(nv)],
                uv=[tuple(x / 4096.0 for x in struct.unpack_from("<2h", b["vuvs"], i * uvstride)) for i in range(nv)],
                skin=[b["vski"][i * 8:i * 8 + 8] for i in range(nv)] if "vski" in b else None,
                idx=struct.unpack(f"<{nt * 3}H", geo_blocks[ioff]["data"][:nt * 6]))


def skin_matrix(entries):
    """Sum of weight x matrix."""
    return tuple(sum(w * m[k] for w, m in entries) for k in range(16))


def read_vertex_weights(raw, palette):
    """[(bone name, weight 0-1)] of 8 skin bytes (4 weights, 4 palette indices)."""
    out = [(palette[raw[4 + k]][0], raw[k] / 255.0) for k in range(4) if raw[k]]
    total = sum(w for _, w in out)
    return [(b, w / total) for b, w in out]


# ---------------------------------------------------------------- export

class Writer:
    def __init__(self):
        self.buf = bytearray()
        self.accessors, self.views = [], []

    def add(self, data, count, kind, comp, minmax=None):
        while len(self.buf) % 4:
            self.buf.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data)})
        self.buf.extend(data)
        acc = {"bufferView": len(self.views) - 1, "componentType": comp, "count": count, "type": kind}
        if minmax:
            acc["min"], acc["max"] = minmax
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def vec3(self, pts, bounds=True):
        mm = ([min(p[k] for p in pts) for k in range(3)], [max(p[k] for p in pts) for k in range(3)]) if bounds else None
        return self.add(b"".join(struct.pack("<3f", *p) for p in pts), len(pts), "VEC3", 5126, mm)


def export_wipers(p3d, pan_path, out, psi=None):
    p3d, out = Path(p3d), Path(out)
    root = load(p3d)
    pan = Pan(pan_path)
    names = bone_table(root)
    nodes = model_nodes(root)
    world = {}
    for n in nodes:
        world.setdefault(n["name"], n["world"])
    par = tree(root)
    skin = {k: v for k, v in read_skin(root).items() if "iper" in k and "_LOD" not in k}
    if not skin:
        raise ValueError(f"{p3d.name}: no wiper skin found")
    anim_names = [names[b] for b in pan.anim]
    joints = []
    for name in anim_names:
        top = par[name][0]
        if top not in anim_names and top not in joints:
            joints.append(top)
    joints += anim_names
    res = resources(root)
    geo = Geometry(root)
    blocks = {bk["offset"]: bk for bk in geo.blocks}
    w = Writer()
    psi_path = Path(psi) if psi else p3d.parent.parent / (p3d.stem + ".PSI")
    mats = read_psi_materials(psi_path)
    gl_nodes, scene, meshes, materials = [], [], [], {}

    def material(name):
        return materials.setdefault(name, len(materials))

    # bones: one node each, rotation and translation; the top ones carry their whole world position
    index = {}
    for j in joints:
        parent = par[j][0]
        m = gl_matrix(par[j][1] if parent in joints else world[j])
        node = {"name": j, "translation": list(m[12:15]), "rotation": rows_quat(rot3(m))}
        index[j] = len(gl_nodes)
        gl_nodes.append(node)
    for j in joints:
        parent = par[j][0]
        if parent in joints:
            gl_nodes[index[parent]].setdefault("children", []).append(index[j])
        else:
            scene.append(index[j])
    ibm = b"".join(struct.pack("<16f", *invert(gl_matrix(world[j]))) for j in joints)
    skin_acc = w.add(ibm, len(joints), "MAT4", 5126)
    for n in nodes:
        if n["name"] not in skin or n["mesh"] is None:
            continue
        palette = skin[n["name"]]
        joint_of = {b: joints.index(b) for b, _ in palette}
        prims = []
        for pi, part in enumerate(res[n["mesh"]]["parts"]):
            d = part_data(blocks, part)
            pos, nor, jw = [], [], []
            for v, nn, raw in zip(d["pos"], d["nor"], d["skin"]):
                weights = read_vertex_weights(raw, palette)
                ib = dict(palette)
                M = skin_matrix([(wt, matmul(ib[b], world[b])) for b, wt in weights])
                pos.append(to_gl(apply(M, v)))
                x = apply(M, nn, 0.0)
                ln = math.sqrt(sum(c * c for c in x)) or 1.0
                nor.append(to_gl([c / ln for c in x]))
                j = [joint_of[b] for b, _ in weights] + [0] * (4 - len(weights))
                wt = [t for _, t in weights] + [0.0] * (4 - len(weights))
                jw.append((j, wt))
            tris = [tuple(d["idx"][k:k + 3]) for k in range(0, len(d["idx"]), 3)]
            agree = 0
            for a, b, c in tris[:3000]:
                e1 = [pos[b][k] - pos[a][k] for k in range(3)]
                e2 = [pos[c][k] - pos[a][k] for k in range(3)]
                fn = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
                agree += 1 if sum(fn[k] * (nor[a][k] + nor[b][k] + nor[c][k]) for k in range(3)) >= 0 else -1
            if agree < 0:
                tris = [(a, c, b) for a, b, c in tris]
            mlist = mats.get(n["name"], [])
            prims.append({"attributes": {
                "POSITION": w.vec3(pos), "NORMAL": w.vec3(nor, False),
                "TEXCOORD_0": w.add(b"".join(struct.pack("<2f", *uv) for uv in d["uv"]), len(pos), "VEC2", 5126),
                "JOINTS_0": w.add(b"".join(struct.pack("<4B", *j) for j, _ in jw), len(pos), "VEC4", 5121),
                "WEIGHTS_0": w.add(b"".join(struct.pack("<4f", *t) for _, t in jw), len(pos), "VEC4", 5126)},
                "indices": w.add(b"".join(struct.pack("<3I", *t) for t in tris), 3 * len(tris), "SCALAR", 5125),
                "material": material(mlist[pi] if pi < len(mlist) else f"PART_{pi}")})
        meshes.append({"name": n["name"], "primitives": prims})
        gl_nodes.append({"name": n["name"], "mesh": len(meshes) - 1, "skin": 0})
        scene.append(len(gl_nodes) - 1)
    for ref in REFERENCE_NODES:                         # bodywork for reference, not read back
        n = next((x for x in nodes if x["name"] == ref and x["mesh"] is not None), None)
        if n is None:
            continue
        prims = []
        for part in res[n["mesh"]]["parts"]:
            d = part_data(blocks, part)
            pos = [to_gl(apply(n["world"], v)) for v in d["pos"]]
            prims.append({"attributes": {"POSITION": w.vec3(pos)},
                          "indices": w.add(struct.pack(f"<{len(d['idx'])}I", *d["idx"]), len(d["idx"]), "SCALAR", 5125)})
        meshes.append({"name": "REF_" + ref, "primitives": prims})
        gl_nodes.append({"name": "REF_" + ref, "mesh": len(meshes) - 1})
        scene.append(len(gl_nodes) - 1)
    scene.sort(key=lambda i: "mesh" not in gl_nodes[i])   # Blender 2.80's importer fails if a bone comes first
    channels, samplers = [], []
    duration = 0.0
    for bone in pan.anim:
        name = names[bone]
        keys = pan.keys[bone]
        K0i = transpose(quat_rows(keys[0][1]))
        L = rot3(par[name][1])
        t_loc = par[name][1][12:15]
        qs = []
        for t, q in keys:
            R = mul3(mul3(quat_rows(q), K0i), L)
            qs.append(rows_quat(rot3(gl_matrix(with_translation(R, t_loc)))))
        for i in range(1, len(qs)):                     # keep the quaternions continuous
            if sum(a * c for a, c in zip(qs[i], qs[i - 1])) < 0:
                qs[i] = [-v for v in qs[i]]
        times = [t for t, _ in keys]
        duration = max(duration, times[-1])
        ti = w.add(b"".join(struct.pack("<f", t) for t in times), len(times), "SCALAR", 5126, ([min(times)], [max(times)]))
        ri = w.add(b"".join(struct.pack("<4f", *q) for q in qs), len(qs), "VEC4", 5126)
        samplers.append({"input": ti, "output": ri, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": index[name], "path": "rotation"}})
    gl = {"asset": {"version": "2.0", "generator": "p3d_wipers.py"}, "scene": 0, "scenes": [{"nodes": scene}],
          "nodes": gl_nodes, "meshes": meshes,
          "materials": [{"name": k} for k in materials],
          "skins": [{"name": "Wipers", "joints": [index[j] for j in joints], "inverseBindMatrices": skin_acc,
                     "skeleton": index[joints[0]]}],
          "animations": [{"name": "WIPERS", "channels": channels, "samplers": samplers}],
          "accessors": w.accessors, "bufferViews": w.views,
          "buffers": [{"byteLength": len(w.buf),
                       "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(w.buf)).decode()}]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gl), encoding="utf-8")
    print(f"{out.name}: {len(skin)} wipers, {len(joints)} bones, animation {duration:.2f} s")


# ---------------------------------------------------------------- glTF reader

COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_gltf(path):
    path = Path(path)
    raw = path.read_bytes()

    def uri(u):
        return base64.b64decode(u.split(",", 1)[1]) if u.startswith("data:") else (path.parent / u).read_bytes()
    if raw[:4] == b"glTF":
        length = struct.unpack_from("<I", raw, 8)[0]
        o, js, bins = 12, None, []
        while o < length:
            clen, ctype = struct.unpack_from("<I4s", raw, o)
            chunk = raw[o + 8:o + 8 + clen]
            if ctype == b"JSON":
                js = json.loads(chunk.decode("utf-8"))
            elif ctype == b"BIN\x00":
                bins.append(chunk)
            o += 8 + clen
        return js, [bins[0] if "uri" not in b else uri(b["uri"]) for b in js.get("buffers", [])]
    js = json.loads(raw.decode("utf-8"))
    return js, [uri(b["uri"]) for b in js.get("buffers", [])]


def accessor(js, buffers, i):
    a = js["accessors"][i]
    v = js["bufferViews"][a["bufferView"]]
    fmt, size = COMP[a["componentType"]]
    n = SIZE[a["type"]]
    stride = v.get("byteStride") or size * n
    base = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    out = []
    for k in range(a["count"]):
        vals = struct.unpack_from("<" + fmt * n, buffers[v["buffer"]], base + k * stride)
        if a.get("normalized"):
            div = {"b": 127, "B": 255, "h": 32767, "H": 65535}[fmt]
            vals = tuple(max(-1.0, x / div) for x in vals)
        out.append(vals if n > 1 else vals[0])
    return out


strip = lambda name: name.split(".")[0] if "." in name and name.rsplit(".", 1)[1].isdigit() else name


class Rig:
    """The nodes of an edited glTF and their animated transforms."""

    def __init__(self, path):
        self.js, self.buffers = load_gltf(path)
        nodes = self.js["nodes"]
        self.parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
        self.by_name = {}
        for i, n in enumerate(nodes):
            self.by_name.setdefault(strip(n.get("name", "")), i)
        self.tracks = {}
        for an in self.js.get("animations", []):
            for ch in an["channels"]:
                t = ch["target"]
                if "node" not in t or t["path"] not in ("translation", "rotation", "scale"):
                    continue
                s = an["samplers"][ch["sampler"]]
                times = accessor(self.js, self.buffers, s["input"])
                key = (t["node"], t["path"])
                if key not in self.tracks or len(times) > len(self.tracks[key][0]):
                    self.tracks[key] = (times, accessor(self.js, self.buffers, s["output"]), s.get("interpolation", "LINEAR"))
        self.cache = {}

    def sample(self, node, path, t):
        tr = self.tracks.get((node, path))
        if not tr:
            return None
        times, vals, how = tr
        if how == "CUBICSPLINE":
            vals = vals[1::3]
        if t <= times[0]:
            return vals[0]
        if t >= times[-1]:
            return vals[-1]
        i = next(k for k in range(1, len(times)) if times[k] >= t)
        if how == "STEP":
            return vals[i - 1]
        f = (t - times[i - 1]) / (times[i] - times[i - 1])
        a, b = vals[i - 1], vals[i]
        if path == "rotation" and sum(x * y for x, y in zip(a, b)) < 0:
            b = [-v for v in b]
        out = [x + (y - x) * f for x, y in zip(a, b)]
        if path == "rotation":
            ln = math.sqrt(sum(v * v for v in out))
            out = [v / ln for v in out]
        return out

    def local(self, node, t):
        n = self.js["nodes"][node]
        if "matrix" in n and not any((node, p) in self.tracks for p in ("translation", "rotation", "scale")):
            return tuple(n["matrix"])
        tr = self.sample(node, "translation", t) or n.get("translation", [0, 0, 0])
        q = self.sample(node, "rotation", t) or n.get("rotation", [0, 0, 0, 1])
        s = self.sample(node, "scale", t) or n.get("scale", [1, 1, 1])
        R = quat_rows(q)
        return (R[0][0] * s[0], R[0][1] * s[0], R[0][2] * s[0], 0.0, R[1][0] * s[1], R[1][1] * s[1], R[1][2] * s[1], 0.0,
                R[2][0] * s[2], R[2][1] * s[2], R[2][2] * s[2], 0.0, tr[0], tr[1], tr[2], 1.0)

    def world(self, node, t):
        key = (node, t)
        if key not in self.cache:
            L = self.local(node, t)
            self.cache[key] = matmul(L, self.world(self.parent[node], t)) if node in self.parent else L
        return self.cache[key]


# ---------------------------------------------------------------- import

def set_world(root, name, target):
    """Gives a node a new world matrix; its children follow."""
    world = {}
    for n in model_nodes(root):
        world.setdefault(n["name"], n["world"])
    for c in root.walk():
        if c.id != b"NoHe":
            continue
        ln = struct.unpack_from("<I", c.data, 2)[0]
        if c.data[6:6 + ln].decode("latin-1") == name:
            d = bytearray(c.data)
            at = 6 + ln + 2
            L = struct.unpack_from("<16f", d, at)
            struct.pack_into("<16f", d, at, *matmul(matmul(target, invert(world[name])), L))
            c.data = bytes(d)
            return
    raise ValueError(f"{name}: node not found")


def skin_bytes(weights, palette_index):
    """8 skin bytes of [(bone, weight)]: up to 4 weights summing to 255, then their palette indices."""
    weights = sorted(weights, key=lambda x: -x[1])[:4]
    total = sum(w for _, w in weights)
    q = [int(round(w / total * 255)) for _, w in weights]
    q[0] += 255 - sum(q)
    q += [0] * (4 - len(q))
    idx = [palette_index[b] for b, _ in weights] + [1] * (4 - len(weights))
    return bytes(q + idx)


def import_wipers(p3d, pan_path, edit, out_dir, psi=None, log=print):
    """Rebuilds the wipers from an edited glTF. Writes the model, WIPERS.PAN and the .PSI in out_dir."""
    p3d, pan_path, out_dir = Path(p3d), Path(pan_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = load(p3d)
    pan = Pan(pan_path)
    names = bone_table(root)
    rig = Rig(edit)
    js = rig.js
    world = {}
    for n in model_nodes(root):
        world.setdefault(n["name"], n["world"])
    par = tree(root)
    anim_names = [names[b] for b in pan.anim]
    missing = [b for b in anim_names if b not in rig.by_name]
    if missing:
        raise ValueError(f"{Path(edit).name}: bones missing: {missing} (keep the armature and the bone names)")
    if not js.get("skins"):
        raise ValueError(f"{Path(edit).name}: no armature (skin) in the file")
    glskin = js["skins"][0]
    jnames = [strip(js["nodes"][j]["name"]) for j in glskin["joints"]]
    # every time a bone or one of its parents has a key at
    times = sorted({t for (node, _), tr in rig.tracks.items() if node in glskin["joints"] for t in tr[0]})
    if not times:
        raise ValueError(f"{Path(edit).name}: no animation on the bones (export with Animation on)")
    t0 = times[0]
    W0 = {j: rig.world(rig.by_name[j], t0) for j in jnames}
    # bones: the game's rotation axes, Blender's position and motion
    new_world = {}
    for j in jnames:
        pos = to_game(W0[j][12:15])
        new_world[j] = with_translation(rot3(world[j]), pos)
    drifted = set()

    def bone_at(j, t):
        d = matmul(invert(W0[j]), rig.world(rig.by_name[j], t))
        return matmul(new_world[j], game_matrix(d))

    def parent_at(j, t):
        p = par[j][0]
        return bone_at(p, t) if p in jnames else world[p]
    new_local = {}
    for j in jnames:
        new_local[j] = matmul(new_world[j], invert(parent_at(j, t0)))
    for bone in pan.anim:
        j = names[bone]
        keys, prev = [], None
        for t in times:
            B = bone_at(j, t)
            L = matmul(B, invert(parent_at(j, t)))
            if math.dist(L[12:15], new_local[j][12:15]) > DRIFT:
                drifted.add(j)
            q = rows_quat(orthonormal(rot3(L)))
            if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
                q = [-v for v in q]
            keys.append((t - t0, q))
            prev = q
        pan.keys[bone] = keys
    if drifted:
        log(f"  warning: these bones change position relative to their parent, which the game cannot animate "
            f"(only rotation is kept): {sorted(drifted)}")
    for k, b in enumerate(pan.trans):
        if names[b] in jnames:
            pan.trans_values[k] = tuple(new_local[names[b]][12:15])
    # meshes
    skin = read_skin(root)
    ibms = accessor(js, rig.buffers, glskin["inverseBindMatrices"])
    jw0 = [rig.world(j, t0) for j in glskin["joints"]]
    lines, nv, nt, nn = [], 0, 0, 0
    stored = {}
    for i, n in enumerate(js["nodes"]):
        name = strip(n.get("name", ""))
        if "mesh" not in n or name not in skin or name.endswith("_LOD1"):
            continue
        palette = skin[name]
        ib = dict(palette)
        pindex = {b: k for k, (b, _) in enumerate(palette)}
        nw = world[name]
        lines.append(f"o {name}")
        pts, skins = [], []
        for prim in js["meshes"][n["mesh"]]["primitives"]:
            at = prim["attributes"]
            if "JOINTS_0" not in at:
                raise ValueError(f"{name}: no bone weights in the file")
            pos = accessor(js, rig.buffers, at["POSITION"])
            nor = accessor(js, rig.buffers, at["NORMAL"])
            uvs = accessor(js, rig.buffers, at["TEXCOORD_0"])
            ji, wi = accessor(js, rig.buffers, at["JOINTS_0"]), accessor(js, rig.buffers, at["WEIGHTS_0"])
            idx = accessor(js, rig.buffers, prim["indices"])
            mat = strip(js["materials"][prim["material"]]["name"]) if "material" in prim else "BODY_NRM"
            lines.append(f"usemtl {mat}")
            for p, q, uv, j4, w4 in zip(pos, nor, uvs, ji, wi):
                pairs = [(jk, wk) for jk, wk in zip(j4, w4) if wk > 1e-4]
                total = sum(wk for _, wk in pairs)
                if not pairs:
                    raise ValueError(f"{name}: a vertex has no bone weight")
                S = skin_matrix([(wk / total, matmul(ibms[jk], jw0[jk])) for jk, wk in pairs])
                rest = to_game(apply(S, p))
                x = apply(S, q, 0.0)
                bad = sorted({jnames[jk] for jk, _ in pairs} - set(pindex))
                if bad:
                    raise ValueError(f"{name}: vertices weighted to {bad}, but this wiper's skin can use only "
                                     f"{sorted(pindex)}")
                combined = {}
                for jk, wk in pairs:
                    combined[jnames[jk]] = combined.get(jnames[jk], 0.0) + wk / total
                Mi = invert(skin_matrix([(wt, matmul(ib[b], new_world[b])) for b, wt in combined.items()]))
                v = apply(Mi, rest)
                wp = apply(nw, v)
                ng = apply(nw, apply(Mi, to_game(x), 0.0), 0.0)
                ln = math.sqrt(sum(c * c for c in ng)) or 1.0
                lines.append("v {:.6f} {:.6f} {:.6f}".format(*to_gl(wp)))
                lines.append(f"vt {uv[0]:.6f} {1 - uv[1]:.6f}")
                lines.append("vn {:.5f} {:.5f} {:.5f}".format(*to_gl([c / ln for c in ng])))
                pts.append(v)
                skins.append(skin_bytes(list(combined.items()), pindex))
            for k in range(0, len(idx), 3):
                t = [idx[k + m] + nv + 1 for m in range(3)]
                lines.append("f " + " ".join(f"{a}/{a}/{a}" for a in t))
            nv += len(pos)
        stored[name] = (pts, skins)
    if not stored:
        raise ValueError(f"{Path(edit).name}: no wiper meshes (objects must keep the names of the wipers)")
    out_p3d = out_dir / p3d.name
    psi_in = Path(psi) if psi else p3d.parent.parent / (p3d.stem + ".PSI")
    psi_out = out_dir / psi_in.name
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "wipers.obj"
        obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
        messages = []
        import_obj(p3d, obj, out_p3d, only=list(stored), same_lod=True, force=True, log=messages.append,
                   psi=psi_in, psi_out=psi_out, allow_skin=True)
    # the bones each vertex hangs from, by position (the import copies them from the nearest original vertex)
    root = load(out_p3d)
    geo = Geometry(root)
    blocks = {bk["offset"]: bk for bk in geo.blocks}
    res = resources(root)
    worst = 0.0
    for n in model_nodes(root):
        base = re.sub(r"_LOD\d+$", "", n["name"])
        if n["mesh"] is None or base not in stored:
            continue
        pts, skins = stored[base]
        near = Nearest(pts)
        for part in res[n["mesh"]]["parts"]:
            count, st, _, _ = Geometry.suge_fields(part["suge"])
            st = dict(st)
            vp = blocks[st["vpos"]]["data"]
            data = bytearray()
            for i in range(count):
                p = struct.unpack_from("<3f", vp, i * 12)
                k = near.find(p)
                worst = max(worst, math.dist(p, pts[k]))
                data += skins[k]
            blocks[st["vski"]]["data"] = bytes(data)
    geo.rebuild()
    for j in jnames:                                     # parents first: jnames is in tree order
        set_world(root, j, new_world[j])
    out_p3d.write_bytes(serialize(root))
    pan_out = out_dir / pan_path.name
    pan.save(pan_out)
    log(f"  {', '.join(stored)}: {nv} vertices, {len(times)} key times, {times[-1] - t0:.2f} s, "
        f"largest skin lookup distance {worst * 1000:.2f} mm")
    return out_p3d, pan_out, psi_out


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "export":
        export_wipers(*a[1:])
    elif len(a) == 5 and a[0] == "import":
        import_wipers(*a[1:])
    else:
        raise SystemExit(__doc__ or "usage: p3d_wipers.py export|import ...")
