# OBJ -> P3D: puts parts edited in Blender back into the original WRC Generations model.
#
#   python obj_to_p3d.py ORIGINAL.P3D EDITED.obj OUTPUT.P3D [--only Spoiler,Hood] [--same-lod] [--force]
#                        [--psi MODEL.PSI]
#
# The OBJ must come from p3d_to_obj.py (same model) and be edited and exported again, e.g. from Blender
# (File > Export > Wavefront OBJ, default axes Y up / -Z forward, with UVs and normals; triangulating is optional).
# Objects are matched to the model's nodes by name: a second node with the same name is "Name.001", as Blender
# names it. Inside an object, faces are grouped by material name (BODY, GLASS...):
#   - a material the node already has replaces that material's parts
#   - a material the node does not have, but the car uses somewhere, becomes a new part of the node
#   - a material of the node with no faces left in the object: its parts are removed
#   - more than 65535 vertices for one material: split into several parts (16-bit indices)
# When parts are added or removed, the model's .PSI (materials per part) changes too: it is written next to
# OUTPUT.P3D, one folder up as in the game (OUTPUT_FOLDER\..\MODEL.PSI), or to --psi-out.
#
#   --only      only these nodes (comma separated); default: every node whose geometry changed
#   --same-lod  also puts the new shape into the node's lower levels of detail (Name_LOD1, Name_LOD2...)
#   --force     rewrite the nodes even when nothing changed (used by the round-trip check)
#   --psi       the model's .PSI, when it is not one folder up from ORIGINAL.P3D as in the game
#   --psi-out   where to write the changed .PSI
#
# Objects Convex_... replace the physics hulls with the same name (see p3d_phac.py).
#
# For each rewritten part: positions, normals and UV0 come from the OBJ. What the OBJ cannot carry (vertex
# colour, tangents, the 4th normal value, UV1/UV2) is copied from the nearest vertex of the node's original
# parts with that material. A new part takes the vertex format and these values from the node's own parts:
# all parts of a mesh resource must share the vertex format (rule of the game files; breaking it made the
# game hang while loading the car). Skinned parts (wipers, suspensions) are refused unless allow_skin=True
# (import_obj only): then the bone indices/weights are also copied from the nearest original vertex, which is
# right for rigid parts bound to one bone (steering wheel: every vertex 255 on bone 0 or 1). Bounding boxes (MRsc)
# are recomputed; the rest of the file stays as it was.

import copy
import math
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3d_lib import Chunk, Geometry, load, serialize  # noqa: E402
from p3d_to_obj import matmul  # noqa: E402
from p3d_psi import Psi  # noqa: E402

IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
LOD = re.compile(r"_LOD\d+$", re.I)
MAX_VERTS = 0xFFFF


# ---------------------------------------------------------------- model side

def model_nodes(root):
    """Nodes in the exporter's order: dict(name, world matrix, mesh resource index or None, occurrence)."""
    out, seen = [], {}

    def walk(lst, parent):
        cur = None
        for c in lst:
            if c.id == b"NoHe":
                n = struct.unpack_from("<I", c.data, 2)[0]
                m = struct.unpack_from("<16f", c.data, 6 + n + 2)
                name = c.data[6:6 + n].decode("latin-1")
                cur = dict(name=name, world=matmul(m, parent), mesh=None)
                out.append(cur)
            elif c.id == b"MeRe":
                cur["mesh"] = struct.unpack_from("<I", c.data, 0)[0]
            elif c.kind == b"Nods":
                walk(c.children, cur["world"] if cur else parent)

    walk(root.find(b"Nods").children, IDENTITY)
    for n in out:      # which node with this name it is, counting only nodes with a mesh (as the .PSI does)
        if n["mesh"] is not None:
            n["occurrence"] = seen.get(n["name"], 0)
            seen[n["name"]] = n["occurrence"] + 1
    return out


def resources(root):
    out, cur = [], None
    for c in root.find(b"MeLi").children:
        if c.id == b"MRsc":
            cur = dict(mrsc=c, parts=[])
            out.append(cur)
        elif c.kind == b"GeLi":
            fmt = next(x for x in c.children if x.id == b"Geom").data
            cur["parts"].append(dict(geli=c, suge=next(x for x in c.walk() if x.id == b"SuGe"),
                                     format=fmt[10:-2].decode()))
    return out


def invert(m):
    """Inverse of an affine row-major 4x4 (row-vector convention)."""
    a = [[m[r * 4 + c] for c in range(3)] for r in range(3)]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    inv = [[(a[(c + 1) % 3][(r + 1) % 3] * a[(c + 2) % 3][(r + 2) % 3]
             - a[(c + 1) % 3][(r + 2) % 3] * a[(c + 2) % 3][(r + 1) % 3]) / det for c in range(3)] for r in range(3)]
    t = [m[12], m[13], m[14]]
    nt = [-sum(t[k] * inv[k][c] for k in range(3)) for c in range(3)]
    return tuple(inv[0] + [0] + inv[1] + [0] + inv[2] + [0] + nt + [1])


def apply(m, v, w=1.0):
    return tuple(v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + w * m[12 + c] for c in range(3))


def material_name(pma):
    return Path(pma.replace("DATA:/", "")).stem.upper()


# ---------------------------------------------------------------- OBJ side

def read_obj(path):
    """{object name: [(material, [(v, vt, vn), ...] triangles)]} keeping the order of materials, and the
    global position / uv / normal lists (game axes: OBJ (x, y, z) = game (x, -z, y))."""
    V, T, N, objs = [], [], [], {}
    cur, mat = None, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            x, y, z = map(float, p[1:4])
            V.append((x, -z, y))
        elif p[0] == "vt":
            T.append((float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
        elif p[0] == "vn":
            x, y, z = map(float, p[1:4])
            N.append((x, -z, y))
        elif p[0] == "o":
            # a second object with the same name (p3d_to_obj writes one per node) becomes Name.001, like Blender
            name, k = line[2:].strip(), 0
            while (name if k == 0 else f"{name}.{k:03d}") in objs:
                k += 1
            cur, mat = objs.setdefault(name if k == 0 else f"{name}.{k:03d}", []), None
        elif p[0] == "usemtl":
            mat = re.sub(r"\.\d{3}$", "", p[1]).upper()     # Blender adds .001 to duplicated materials
        elif p[0] == "f":
            if cur is None:
                cur = objs.setdefault("", [])
            if not cur or cur[-1][0] != mat:
                cur.append((mat, []))
            idx = []
            for item in p[1:]:
                f = item.split("/")
                fix = lambda s, n: (int(s) - 1 if int(s) > 0 else n + int(s)) if s else None
                idx.append((fix(f[0], len(V)), fix(f[1], len(T)) if len(f) > 1 else None,
                            fix(f[2], len(N)) if len(f) > 2 else None))
            for k in range(1, len(idx) - 1):          # fan triangulation of quads and polygons
                cur[-1][1].append((idx[0], idx[k], idx[k + 1]))
    return objs, V, T, N


def node_key(name):
    """'Needle_RPM.001' -> ('Needle_RPM', 1)."""
    m = re.match(r"^(.*)\.(\d{3})$", name)
    return (m[1], int(m[2])) if m else (name, 0)


# ---------------------------------------------------------------- vertex data

class Nearest:
    """Nearest original vertex by position: k-d tree, fast also for points far from the original ones
    (a new part placed away from the parts it copies its vertex data from)."""

    def __init__(self, pts):
        self.pts = pts
        self.root = self._build(list(range(len(pts))), 0)

    def _build(self, idx, depth):
        if len(idx) <= 8:
            return idx                         # leaf: list of point indices
        ax = depth % 3
        idx.sort(key=lambda i: self.pts[i][ax])
        mid = len(idx) // 2
        return (ax, self.pts[idx[mid]][ax], self._build(idx[:mid], depth + 1), self._build(idx[mid:], depth + 1))

    def find(self, p):
        best = [float("inf"), 0]

        def visit(node):
            if isinstance(node, list):
                for i in node:
                    q = self.pts[i]
                    d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
                    if d < best[0]:
                        best[0], best[1] = d, i
                return
            ax, split_at, lo, hi = node
            diff = p[ax] - split_at
            near, far = (lo, hi) if diff < 0 else (hi, lo)
            visit(near)
            if diff * diff < best[0]:
                visit(far)

        visit(self.root)
        return best[1]


def part_streams(blocks, suge):
    nv, streams, ntri, ioff = Geometry.suge_fields(suge)
    return nv, [(name, blocks[off]) for name, off in streams], ntri, blocks[ioff]


def winding_agrees(pos, nor, idx):
    """Same test as the exporter: True if triangle order agrees with the vertex normals."""
    agree = 0
    for t in range(0, min(len(idx), 9000), 3):
        a, b, c = pos[idx[t]], pos[idx[t + 1]], pos[idx[t + 2]]
        e1 = [b[k] - a[k] for k in range(3)]
        e2 = [c[k] - a[k] for k in range(3)]
        fn = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        vn = [nor[idx[t]][k] + nor[idx[t + 1]][k] + nor[idx[t + 2]][k] for k in range(3)]
        agree += 1 if sum(fn[k] * vn[k] for k in range(3)) >= 0 else -1
    return agree >= 0


class Pool:
    """Original vertices of one or more parts with the same material and format, in one node's local space:
    source of everything the OBJ does not carry."""

    def __init__(self, parts, blocks, to_local):
        """parts: [(part dict, matrix from that part's node space to the target node space)]"""
        self.format = parts[0][0]["format"]
        self.template = parts[0][0]
        self.names, self.strides = [], {}
        self.pos, self.nor, self.raw = [], [], {}
        flips = []
        for part, m in parts:
            if part["format"] != self.format:
                continue
            nv, streams, ntri, iblk = part_streams(blocks, part["suge"])
            if not self.names:
                self.names = [n for n, b in streams]
            sd = dict(streams)
            pos = [struct.unpack_from("<3f", sd["vpos"]["data"], i * 12) for i in range(nv)]
            nor = [struct.unpack_from("<4e", sd["vnor"]["data"], i * 8) for i in range(nv)]
            flips.append(not winding_agrees(pos, nor, struct.unpack_from(f"<{ntri * 3}H", iblk["data"])))
            self.pos += [apply(m, p) for p in pos]
            self.nor += nor
            for name, blk in streams:
                stride = len(blk["data"]) // nv
                self.strides[name] = stride
                self.raw.setdefault(name, bytearray()).extend(blk["data"])
        self.flip = sum(flips) * 2 > len(flips)      # the exporter flipped these parts: flip back
        self.uv_count = len(re.findall(r"UV\d_", self.format)) or 1
        self.near = Nearest(self.pos)


def make_part(tris, V, T, N, inv, pool):
    """Stream data [(name, bytes)] in the pool's order, index bytes, vertex count, local positions."""
    verts, index, new_idx = [], {}, []
    for tri in tris:
        corners = []
        for c in tri:
            if c not in index:
                index[c] = len(verts)
                verts.append(c)
            corners.append(index[c])
        if pool.flip:
            corners[1], corners[2] = corners[2], corners[1]
        new_idx += corners
    pos, nor, uv, src = [], [], [], []
    for vi, ti, ni in verts:
        p = apply(inv, V[vi])
        n = apply(inv, N[ni], 0.0) if ni is not None else (0.0, 0.0, 1.0)
        ln = math.sqrt(sum(x * x for x in n)) or 1.0
        pos.append(p)
        nor.append(tuple(x / ln for x in n))
        uv.append(T[ti] if ti is not None else (0.0, 0.0))
        src.append(pool.near.find(p))
    clamp = lambda x: max(-32768, min(32767, int(round(x * 4096))))
    streams = []
    for name in pool.names:
        old, stride = pool.raw[name], pool.strides[name]
        if name == "vpos":
            data = b"".join(struct.pack("<3f", *p) for p in pos)
        elif name == "vnor":
            data = b"".join(struct.pack("<4e", *n, pool.nor[s][3]) for n, s in zip(nor, src))
        elif name == "vuvs":
            out = bytearray()
            for (u, v), s in zip(uv, src):
                out += struct.pack("<2h", clamp(u), clamp(1.0 - v))
                out += old[s * stride + 4:s * stride + 4 * pool.uv_count]    # UV1/UV2 from the nearest vertex
            data = bytes(out)
        else:                                       # vcol, vtan and anything unknown
            data = b"".join(bytes(old[s * stride:(s + 1) * stride]) for s in src)
        streams.append((name, data))
    return streams, struct.pack(f"<{len(new_idx)}H", *new_idx), len(verts), pos


def split(tris):
    """Triangle lists with at most MAX_VERTS distinct vertices each."""
    out, cur, seen = [], [], set()
    for t in tris:
        new = {c for c in t if c not in seen}
        if len(seen) + len(new) > MAX_VERTS:
            out.append(cur)
            cur, seen, new = [], set(), set(t)
        cur.append(t)
        seen |= new
    if cur:
        out.append(cur)
    return out


def set_suge(suge, nv, ntri):
    d = bytearray(suge.data)
    ns = struct.unpack_from("<I", d, 6)[0]
    struct.pack_into("<I", d, 2, nv)
    struct.pack_into("<I", d, 10 + ns * 8, ntri)
    suge.data = bytes(d)


MRSC_COUNT = 46     # offset of the part count after the tag: box 24 + CD AB + u16 1 + sphere 16 + CD AB


def set_mrsc(mrsc, pts, count=None):
    """MRsc: u16 6, u32 length + tag (MESH/SKINMESH), u16 1, box min/max (6 floats), CD AB, u16 1,
    sphere centre + radius (4 floats), CD AB, u32 number of parts (geometries) of the mesh, CD AB.
    The game takes the number of geometries from here, not from the GeLi lists that follow."""
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    d = bytearray(mrsc.data)
    at = 6 + struct.unpack_from("<I", d, 2)[0] + 2
    struct.pack_into("<6f", d, at, *lo, *hi)
    struct.pack_into("<4f", d, at + 28, *[(a + b) / 2 for a, b in zip(lo, hi)],
                     math.sqrt(sum(((b - a) / 2) ** 2 for a, b in zip(lo, hi))))
    if count is not None:
        struct.pack_into("<I", d, at + MRSC_COUNT, count)
    mrsc.data = bytes(d)


def mrsc_count(mrsc):
    d = mrsc.data
    return struct.unpack_from("<I", d, 6 + struct.unpack_from("<I", d, 2)[0] + 2 + MRSC_COUNT)[0]


# ---------------------------------------------------------------- change detection

def node_corners_by_material(node, parts, mats, blocks):
    """Original triangle corners of a node in world space, per material."""
    out = {}
    for part, mat in zip(parts, mats):
        nv, streams, ntri, iblk = part_streams(blocks, part["suge"])
        sd = dict(streams)
        pos = [apply(node["world"], struct.unpack_from("<3f", sd["vpos"]["data"], i * 12)) for i in range(nv)]
        out.setdefault(mat, []).extend(pos[i] for i in struct.unpack_from(f"<{ntri * 3}H", iblk["data"]))
    return out


def same_shape(a, b, tol=2e-4):
    """Same corner positions (any order) within tol: the OBJ keeps 6 decimals and Blender may change the last one."""
    if len(a) != len(b):
        return False
    key = lambda p: tuple(round(c, 2) for c in p)
    a, b = sorted(a, key=key), sorted(b, key=key)
    if all(max(abs(x - y) for x, y in zip(p, q)) <= tol for p, q in zip(a, b)):
        return True
    grid = {}
    for q in b:
        grid.setdefault(tuple(int(math.floor(c / 0.01)) for c in q), []).append(q)
    for p in a:
        k = tuple(int(math.floor(c / 0.01)) for c in p)
        if not any(max(abs(x - y) for x, y in zip(p, q)) <= tol
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   for q in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ())):
            return False
    return True


# ---------------------------------------------------------------- physics hulls

def import_hulls(root, hull_objs, V, force):
    """Physics hulls from OBJ objects named like the PhAc records (Convex_...). The new hull is the convex
    hull of the object's vertices. Its record id is the CRC32 of the cooked data: in the game equal data
    always have equal ids, and this keeps that true for new hulls (the game's own formula is not known;
    the game accepted these ids in a test)."""
    import zlib
    from p3d_phac import build_hull, read_phac, write_hull, write_phac
    chunk = next((c for c in root.walk() if c.id == b"PhAc"), None)
    if chunk is None:
        raise ValueError("the model has no physics hulls to replace")
    ph = read_phac(chunk.data)
    names = {r.name.decode("latin-1"): r for r in ph.records}
    done = []
    for name, groups in hull_objs.items():
        r = names.get(name)
        if r is None:
            raise ValueError(f"'{name}': no physics hull with this name (new hulls cannot be added yet)")
        inv = invert(r.matrix)
        idx = sorted({c[0] for mat, tris in groups for t in tris for c in t})
        pts = [apply(inv, V[i]) for i in idx]
        if not force and same_shape(pts, r.hull.vertices, 2e-4):
            continue
        r.hull = build_hull(pts)
        r.id = zlib.crc32(write_hull(r.hull))
        done.append(name)
    if done:
        chunk.data = write_phac(ph)
    return done


# ---------------------------------------------------------------- whole model

class Model:
    def __init__(self, p3d, psi_path=None, allow_skin=False):
        self.allow_skin = allow_skin
        self.root = load(p3d)
        self.geo = Geometry(self.root)
        self.blocks = {b["offset"]: b for b in self.geo.blocks}
        self.res = resources(self.root)
        self.nodes = model_nodes(self.root)
        psi_path = Path(psi_path) if psi_path else Path(p3d).parent.parent / (Path(p3d).stem + ".PSI")
        self.psi = Psi(psi_path) if psi_path.exists() else None
        self.psi_changed = False
        self.users = {}
        for n in self.nodes:
            if n["mesh"] is not None:
                self.users.setdefault(n["mesh"], []).append(n)

    def materials(self, node):
        """Material names of the node's parts, in part order."""
        parts = self.res[node["mesh"]]["parts"]
        mats = [material_name(p) for p, f in self.psi.materials(node["name"], node["occurrence"])] if self.psi else []
        if len(mats) != len(parts):
            mats = [f"PART_{node['mesh']}_{i}" for i in range(len(parts))]
        return mats

    def material_parts(self, mat):
        """(part, node) with this material anywhere in the model."""
        out = []
        for n in self.nodes:
            if n["mesh"] is None:
                continue
            for part, m in zip(self.res[n["mesh"]]["parts"], self.materials(n)):
                if m == mat:
                    out.append((part, n))
        return out

    def rebuild_node(self, node, mat_tris, V, T, N, log):
        """Replaces the node's parts with the OBJ triangles grouped by material (see the top of the file)."""
        res = self.res[node["mesh"]]
        parts = res["parts"]
        mats = self.materials(node)
        flags = [f for p, f in self.psi.materials(node["name"], node["occurrence"])] if self.psi else [0] * len(parts)
        if len(flags) != len(parts):
            flags = [0] * len(parts)
        if any("SKIN" in p["format"] for p in parts) and not self.allow_skin:
            raise ValueError(f"'{node['name']}': skinned part (bones): not supported yet "
                             "(allow_skin: bone weights copied from the nearest original vertex)")
        inv = invert(node["world"])
        meli = self.root.find(b"MeLi")
        keep, new_parts, allpts = [], [], []
        for mat in list(dict.fromkeys(mats + list(mat_tris))):
            tris = mat_tris.get(mat, [])
            own = [i for i, m in enumerate(mats) if m == mat]
            if own:
                pool = Pool([(parts[i], IDENTITY) for i in own], self.blocks, inv)
            else:
                # In the game every part of a mesh resource has the same vertex format (4014 resources, no
                # exception; a new part with another format made the game hang while loading the car).
                # So a new part copies the format and the vertex data of the node's own parts: only its
                # material (.PSI entry) is new.
                others = self.material_parts(mat)
                if not others or self.psi is None or self.psi.pma_for(mat) is None:
                    raise ValueError(f"'{node['name']}': material {mat} is not used anywhere in this car "
                                     "(a new material needs its own .PMA: not supported yet)")
                usual = {p["format"] for p, n in others}
                if parts[0]["format"] not in usual:
                    log(f"  '{node['name']}': new {mat} part gets this node's vertex format {parts[0]['format']} "
                        f"(elsewhere {mat} uses {sorted(usual)})")
                pool = Pool([(p, IDENTITY) for p in parts], self.blocks, inv)
            chunks = split(tris) if tris else []
            if not chunks and own:
                log(f"  '{node['name']}': no faces left with {mat}, its part(s) removed")
            for k, ch in enumerate(chunks):
                streams, idx, nv, pts = make_part(ch, V, T, N, inv, pool)
                allpts += pts
                if k < len(own):                       # reuse an existing part of this material
                    part = parts[own[k]]
                    _, old, _, iblk = part_streams(self.blocks, part["suge"])
                    if [n for n, b in old] != [n for n, d in streams]:
                        raise ValueError("stream order changed")
                    for (name, blk), (_, data) in zip(old, streams):
                        blk["data"] = data
                    iblk["data"] = idx
                    set_suge(part["suge"], nv, len(idx) // 6)
                    keep.append((own[k], part, mat, flags[own[k]]))
                else:                                  # new part, copied from the pool's first part
                    geli = copy.deepcopy(pool.template["geli"])
                    suge = next(x for x in geli.walk() if x.id == b"SuGe")
                    set_suge(suge, nv, len(idx) // 6)
                    self.geo.add_part(suge, streams, idx)
                    part = dict(geli=geli, suge=suge, format=pool.format)
                    new_parts.append((part, mat))
            for i in own[len(chunks):]:               # parts of this material no longer needed
                self.geo.drop_part(parts[i]["suge"])
                meli.children.remove(parts[i]["geli"])
        if not keep and not new_parts:
            raise ValueError(f"'{node['name']}': no faces at all")
        # new parts go right after the node's last part in MeLi
        keep.sort(key=lambda x: x[0])
        last = max((meli.children.index(p["geli"]) for _, p, _, _ in keep), default=None)
        if last is None:
            last = meli.children.index(res["mrsc"])
        for k, (part, mat) in enumerate(new_parts):
            meli.children.insert(last + 1 + k, part["geli"])
        res["parts"] = [p for _, p, _, _ in keep] + [p for p, m in new_parts]
        set_mrsc(res["mrsc"], allpts, len(res["parts"]))
        new_mats = [m for _, _, m, _ in keep] + [m for p, m in new_parts]
        if new_mats != mats:
            if self.psi is None:
                raise ValueError(f"'{node['name']}': parts added or removed, the model's .PSI is needed (--psi)")
            pmas = [(self.psi.pma_for(m), f) for m, f in zip(new_mats, [f for _, _, _, f in keep] + [0] * len(new_parts))]
            self.psi.set_materials(node["name"], node["occurrence"], pmas)
            self.psi_changed = True
        return new_mats


def import_obj(p3d, obj, out, only=None, same_lod=False, force=False, log=print, psi=None, psi_out=None,
               allow_skin=False):
    model = Model(p3d, psi, allow_skin)
    by_name = {}
    for n in model.nodes:
        if n["mesh"] is not None:
            by_name.setdefault(n["name"], []).append(n)
    objs, V, T, N = read_obj(obj)
    done, hull_objs = [], {}
    for oname, groups in objs.items():
        base, k = node_key(oname)
        if only and base not in only:
            continue
        if base.startswith("Convex_"):
            hull_objs[base] = groups      # physics hull: handled after the meshes
            continue
        cand = by_name.get(base, [])
        if k >= len(cand):
            log(f"  skipped '{oname}': no such node in the model")
            continue
        node = cand[k]
        if len(model.users[node["mesh"]]) > 1:
            raise ValueError(f"'{oname}': its mesh is shared with {[u['name'] for u in model.users[node['mesh']]]}")
        mat_tris = {}
        for mat, tris in groups:
            mat_tris.setdefault(mat, []).extend(tris)
        mats = model.materials(node)
        if not force and set(mat_tris) == set(mats):
            orig = node_corners_by_material(node, model.res[node["mesh"]]["parts"], mats, model.blocks)
            if all(same_shape([V[c[0]] for t in mat_tris[m] for c in t], orig[m]) for m in orig):
                continue
        targets = [node]
        if same_lod:
            for lod in sorted(n for n in by_name if re.fullmatch(re.escape(base) + r"_LOD\d+", n, re.I)):
                if k < len(by_name[lod]):
                    targets.append(by_name[lod][k])
        for t in targets:
            before = model.materials(t)
            after = model.rebuild_node(t, mat_tris, V, T, N, log)
            done.append(t["name"])
            if after != before:
                log(f"  '{t['name']}': parts {before} -> {after}")
    if hull_objs:
        done += import_hulls(model.root, hull_objs, V, force)
    for i, r in enumerate(model.res):     # rules of every game file, checked before writing
        if len({p["format"] for p in r["parts"]}) > 1:
            raise ValueError(f"mesh resource {i} would mix vertex formats {sorted({p['format'] for p in r['parts']})}")
        if mrsc_count(r["mrsc"]) != len(r["parts"]):
            raise ValueError(f"mesh resource {i}: part count {mrsc_count(r['mrsc'])} in MRsc, {len(r['parts'])} parts")
    model.geo.rebuild()
    Path(out).write_bytes(serialize(model.root))
    if model.psi_changed:
        dest = Path(psi_out) if psi_out else Path(out).parent.parent / (Path(out).stem + ".PSI")
        if dest.resolve() == (Path(psi) if psi else Path(p3d).parent.parent / (Path(p3d).stem + ".PSI")).resolve():
            raise ValueError(f"the changed .PSI would overwrite the original {dest}: give --psi-out")
        dest.write_bytes(model.psi.to_bytes())
        log(f"{dest.name}: written (parts added or removed)")
    log(f"{Path(out).name}: rewritten nodes {done if done else 'none'}")
    return done


USAGE = ("usage: obj_to_p3d.py ORIGINAL.P3D EDITED.obj OUTPUT.P3D [--only A,B] [--same-lod] [--force] "
         "[--psi MODEL.PSI] [--psi-out NEW.PSI]")


def main():
    args, opts, it = [], {}, iter(sys.argv[1:])
    for a in it:
        if a in ("--only", "--psi", "--psi-out"):
            opts[a] = next(it, None)
            if opts[a] is None:
                raise SystemExit(USAGE)
        elif a in ("--same-lod", "--force"):
            opts[a] = True
        else:
            args.append(a)
    if len(args) != 3:
        raise SystemExit(USAGE)
    if Path(args[2]).resolve() == Path(args[0]).resolve():
        raise SystemExit("the output must be a new file, not the original")
    only = set(opts["--only"].split(",")) if "--only" in opts else None
    import_obj(args[0], args[1], args[2], only, "--same-lod" in opts, "--force" in opts,
               psi=opts.get("--psi"), psi_out=opts.get("--psi-out"))


if __name__ == "__main__":
    main()
