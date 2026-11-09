# Physics hulls of a P3D (PhAc chunk): read, write, and build new convex hulls.
#
# PhAc: u16 3, u16 0, u16 0, u32 record count, records, CD AB. A convex record (type 4):
#   u32 length + name, u32 4, 16 floats (4x4 matrix), u32 0, u16 1, u32 id, u32 1, u32 size,
#   PhysX cooked convex mesh (size bytes), CD AB
# The id is the same for equal cooked data everywhere in the game (1092 hulls, 981 distinct); how it is
# computed is not known.
#
# Cooked convex mesh = PhysX 4 ConvexMesh, serial version 13 (little-endian):
#   "NXS\x01" "CVXM" u32 13 u32 0 | "ICE\x01" "CLHL" u32 8 | "ICE\x01" "CVHL" u32 8
#   u32 vertices, edges, polygons, polygon vertex refs
#   vertices (3 floats each)
#   polygons, 20 bytes each: plane (normal xyz, d) 4 floats, u16 offset in the vertex refs, u8 vertex count,
#                            u8 index of the hull vertex with the lowest projection on the normal
#   u8 vertex refs, u8 2 polygons per edge, u8 3 polygons per vertex
#   float geometric epsilon, 6 floats bounds (min, max), float mass (= volume), 9 floats inertia tensor
#   (for density 1, about the centre of mass), 3 floats centre of mass, float -1 (no big convex data) or
#   1 + big convex data (hulls with more than 32 vertices), float internal radius, 3 floats internal extents

import struct
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Hull:
    vertices: list
    polygons: list            # (plane 4 floats, vref, count, min_index)
    vertex_refs: bytes
    faces_by_edges: bytes
    faces_by_vertices: bytes
    epsilon: float
    bounds: tuple
    mass: float
    inertia: tuple
    com: tuple
    big: Optional[bytes]      # big convex data, kept as it is (None if absent)
    radius: float
    extents: tuple
    edges: int = 0
    header: bytes = b""       # the 40 header bytes, kept for exact writing


@dataclass
class Record:
    name: bytes
    matrix: tuple
    id: int
    hull: Optional[Hull] = None
    raw: bytes = b""          # cooked data when it could not be decoded
    kind: int = 4


@dataclass
class PhAc:
    head: bytes
    records: List[Record] = field(default_factory=list)
    tail: bytes = b"\xcd\xab"     # end marker; some models have one more zero byte after it


def read_hull(c):
    head = c[:40]
    if head[:8] != b"NXS\x01CVXM" or head[16:24] != b"ICE\x01CLHL" or head[28:36] != b"ICE\x01CVHL":
        raise ValueError("not a cooked convex mesh")
    nv, ne, npoly, nref = struct.unpack_from("<4I", c, 40)
    q = 56
    verts = [struct.unpack_from("<3f", c, q + i * 12) for i in range(nv)]
    q += nv * 12
    polys = []
    for i in range(npoly):
        plane = struct.unpack_from("<4f", c, q)
        vref, cnt, mi = struct.unpack_from("<HBB", c, q + 16)
        polys.append((plane, vref, cnt, mi))
        q += 20
    refs = c[q:q + nref]; q += nref
    fbe = c[q:q + ne * 2]; q += ne * 2
    fbv = c[q:q + nv * 3]; q += nv * 3
    eps = struct.unpack_from("<f", c, q)[0]; q += 4
    bounds = struct.unpack_from("<6f", c, q); q += 24
    mass = struct.unpack_from("<f", c, q)[0]; q += 4
    inertia = struct.unpack_from("<9f", c, q); q += 36
    com = struct.unpack_from("<3f", c, q); q += 12
    flag = struct.unpack_from("<f", c, q)[0]; q += 4
    big = None
    if flag == 1.0:
        big = c[q:len(c) - 16]
        q = len(c) - 16
    elif flag != -1.0:
        raise ValueError(f"unexpected big convex flag {flag}")
    radius = struct.unpack_from("<f", c, q)[0]
    extents = struct.unpack_from("<3f", c, q + 4)
    if q + 16 != len(c):
        raise ValueError(f"{len(c) - q - 16} unread bytes in the hull")
    return Hull(verts, polys, refs, fbe, fbv, eps, bounds, mass, inertia, com, big, radius, extents, ne, head)


def write_hull(h):
    head = h.header or (b"NXS\x01CVXM" + struct.pack("<II", 13, 0) + b"ICE\x01CLHL" + struct.pack("<I", 8)
                        + b"ICE\x01CVHL" + struct.pack("<I", 8))
    out = bytearray(head)
    out += struct.pack("<4I", len(h.vertices), len(h.faces_by_edges) // 2, len(h.polygons), len(h.vertex_refs))
    for v in h.vertices:
        out += struct.pack("<3f", *v)
    for plane, vref, cnt, mi in h.polygons:
        out += struct.pack("<4fHBB", *plane, vref, cnt, mi)
    out += h.vertex_refs + h.faces_by_edges + h.faces_by_vertices
    out += struct.pack("<f6ff9f3f", h.epsilon, *h.bounds, h.mass, *h.inertia, *h.com)
    if h.big is None:
        out += struct.pack("<f", -1.0)
    else:
        out += struct.pack("<f", 1.0) + h.big
    out += struct.pack("<f3f", h.radius, *h.extents)
    return bytes(out)


# ---------------------------------------------------------------- building a new hull from points
#
# Rules taken from the game's hulls (checked by rebuilding them from their own vertices):
#   polygons wind counter-clockwise seen from outside, plane = (outward normal, d) with n.v + d = 0
#   vertices are numbered in the order they first appear in the polygons
#   min_index = the hull vertex with the lowest n.v
#   faces_by_vertices = the first 3 polygons (in polygon order) that use the vertex
#   mass = volume, inertia = tensor for density 1 about the hull's ORIGIN (not the centre of mass), with
#   off-diagonal terms = minus the products of inertia
#   internal radius = smallest distance from the centre of mass to a plane; internal extents: radius/sqrt(3)
#   on the two smaller axes, and on the largest axis the longest box that still fits inside

MAX_VERTICES = 32      # above this PhysX adds "big convex" data, which is not written here


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    ln = _dot(a, a) ** 0.5
    return (a[0] / ln, a[1] / ln, a[2] / ln) if ln > 0 else (0.0, 0.0, 0.0)


def convex_triangles(pts, eps=1e-6):
    """Incremental 3D convex hull: list of outward counter-clockwise triangles (indices into pts)."""
    n = len(pts)
    if n < 4:
        raise ValueError("a hull needs at least 4 points")
    scale = max(max(abs(c) for c in p) for p in pts) or 1.0
    eps *= scale
    # starting tetrahedron from extreme, non-degenerate points
    i0 = min(range(n), key=lambda i: pts[i])
    i1 = max(range(n), key=lambda i: _dot(_sub(pts[i], pts[i0]), _sub(pts[i], pts[i0])))
    i2 = max(range(n), key=lambda i: _dot(_cross(_sub(pts[i1], pts[i0]), _sub(pts[i], pts[i0])),
                                           _cross(_sub(pts[i1], pts[i0]), _sub(pts[i], pts[i0]))))
    nrm = _cross(_sub(pts[i1], pts[i0]), _sub(pts[i2], pts[i0]))
    i3 = max(range(n), key=lambda i: abs(_dot(nrm, _sub(pts[i], pts[i0]))))
    if abs(_dot(nrm, _sub(pts[i3], pts[i0]))) <= eps * (_dot(nrm, nrm) ** 0.5):
        raise ValueError("the points are flat: no volume")
    faces = [(i0, i1, i2), (i0, i2, i3), (i0, i3, i1), (i1, i3, i2)]
    if _dot(nrm, _sub(pts[i3], pts[i0])) > 0:
        faces = [(a, c, b) for a, b, c in faces]

    def plane(f):
        a, b, c = (pts[i] for i in f)
        nn = _norm(_cross(_sub(b, a), _sub(c, a)))
        return nn, -_dot(nn, a)

    planes = {f: plane(f) for f in faces}
    edge_face = {}
    for f in faces:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_face[e] = f
    # farthest points first: the hull grows quickly and the small, almost flat steps come last
    order = sorted((p for p in range(n) if p not in (i0, i1, i2, i3)),
                   key=lambda p: -max(_dot(planes[f][0], pts[p]) + planes[f][1] for f in faces))
    for p in order:
        dist = {f: _dot(planes[f][0], pts[p]) + planes[f][1] for f in planes}
        best = max(dist, key=dist.get)
        if dist[best] <= eps:
            continue
        # visible region = faces reachable from the most visible one (a connected patch, so the
        # horizon is a single closed loop even when rounding makes far faces look visible)
        vis, stack = {best}, [best]
        while stack:
            f = stack.pop()
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                g = edge_face.get((b, a))
                if g is not None and g not in vis and dist[g] > eps:
                    vis.add(g)
                    stack.append(g)
        horizon = []
        for f in vis:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                if edge_face.get((b, a)) not in vis:
                    horizon.append((a, b))
        for f in vis:
            del planes[f]
            for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                if edge_face.get(e) == f:
                    del edge_face[e]
        for a, b in horizon:
            f = (a, b, p)
            planes[f] = plane(f)
            for e in ((a, b), (b, p), (p, a)):
                edge_face[e] = f
    out = list(planes)
    for f in out:      # every edge must have exactly one opposite edge
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if (b, a) not in edge_face:
                raise ValueError("convex hull is not closed (degenerate points)")
    return out


def _polygons(pts, tris, plane_tol=0.0007):
    """Neighbouring triangles that lie on the same plane (within plane_tol x hull size) merged into polygons:
    [(normal, d, [vertex loop])]. Vertices left on fewer than 3 polygons are removed from the loops."""
    size = max(max(p[k] for p in pts) - min(p[k] for p in pts) for k in range(3))
    tol = plane_tol * size
    info = []
    for t in tris:
        a, b, c = (pts[i] for i in t)
        cr = _cross(_sub(b, a), _sub(c, a))
        info.append((_norm(cr), _dot(cr, cr) ** 0.5))
    nbr = {}
    for ti, t in enumerate(tris):
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            nbr[e] = ti
    group = [None] * len(tris)
    groups = []
    for seed in sorted(range(len(tris)), key=lambda i: -info[i][1]):    # biggest triangles first
        if group[seed] is not None:
            continue
        gi, members = len(groups), [seed]
        group[seed] = gi
        nn = info[seed][0]
        d = -_dot(nn, pts[tris[seed][0]])
        k = 0
        while k < len(members):
            t = tris[members[k]]
            k += 1
            for e in ((t[1], t[0]), (t[2], t[1]), (t[0], t[2])):
                o = nbr.get(e)
                if o is None or group[o] is not None:
                    continue
                if _dot(info[o][0], nn) > 0.999 and all(abs(_dot(nn, pts[v]) + d) <= tol for v in tris[o]):
                    group[o] = gi
                    members.append(o)
        # plane of the group: area-weighted normal, d through the mean of its points
        sx = [sum(info[m][0][j] * info[m][1] for m in members) for j in range(3)]
        nn = _norm(tuple(sx))
        vs = {v for m in members for v in tris[m]}
        d = -sum(_dot(nn, pts[v]) for v in vs) / len(vs)
        groups.append((nn, d, members))
    loops = []
    for nn, d, members in groups:
        directed = set()
        for m in members:
            t = tris[m]
            directed |= {(t[0], t[1]), (t[1], t[2]), (t[2], t[0])}
        border = {a: b for (a, b) in directed if (b, a) not in directed}
        start = min(border)
        loop, v = [start], border[start]
        while v != start:
            loop.append(v)
            v = border[v]
            if len(loop) > len(border):
                raise ValueError("could not close a polygon outline")
        if len(loop) != len(border):
            raise ValueError("a merged polygon has a hole")
        loops.append(loop)
    while True:            # a vertex on only 2 polygons sits on an edge between them: drop it
        count = {}
        for lp in loops:
            for v in lp:
                count[v] = count.get(v, 0) + 1
        drop = {v for v, n in count.items() if n < 3}
        if not drop:
            break
        loops = [[v for v in lp if v not in drop] for lp in loops]
        if any(len(lp) < 3 for lp in loops):
            raise ValueError("a polygon lost its corners while cleaning the hull")
    return [(nn, -sum(_dot(nn, pts[v]) for v in lp) / len(lp), lp) for (nn, d, m), lp in zip(groups, loops)]


def _mass_properties(pts, polys):
    """Volume, centre of mass and inertia tensor about the origin (density 1) from the outward polygons."""
    ref = (0.0, 0.0, 0.0)
    vol, c = 0.0, [0.0, 0.0, 0.0]
    xx = yy = zz = xy = yz = zx = 0.0
    for nn, d, loop in polys:
        a = _sub(pts[loop[0]], ref)
        for k in range(1, len(loop) - 1):
            b, cc = _sub(pts[loop[k]], ref), _sub(pts[loop[k + 1]], ref)
            v6 = _dot(a, _cross(b, cc))            # 6 x volume of tetra (ref, a, b, c)
            vol += v6 / 6
            for j in range(3):
                c[j] += v6 / 24 * (a[j] + b[j] + cc[j])
            # second moments of the tetrahedron with one vertex at the origin
            f = lambda i, j: (a[i] * a[j] + b[i] * b[j] + cc[i] * cc[j]
                              + (a[i] + b[i] + cc[i]) * (a[j] + b[j] + cc[j])) * v6 / 120
            xx += f(0, 0); yy += f(1, 1); zz += f(2, 2); xy += f(0, 1); yz += f(1, 2); zx += f(2, 0)
    if vol <= 0:
        raise ValueError("hull has no volume")
    cm = [x / vol for x in c]
    inertia = (yy + zz, -xy, -zx, -xy, xx + zz, -yz, -zx, -yz, xx + yy)
    com = tuple(cm[k] + ref[k] for k in range(3))
    return vol, com, inertia


def _check_closed(polys):
    owner = {}
    for pi, (nn, d, lp) in enumerate(polys):
        for k in range(len(lp)):
            owner.setdefault(tuple(sorted((lp[k], lp[(k + 1) % len(lp)]))), []).append(pi)
    if any(len(v) != 2 for v in owner.values()):
        raise ValueError("hull is not closed")
    count = {}
    for nn, d, lp in polys:
        for v in lp:
            count[v] = count.get(v, 0) + 1
    if any(n < 3 for n in count.values()):
        raise ValueError("a hull vertex is on fewer than 3 polygons")


def build_hull(points, inertia_sign=1.0):
    """Hull (PhysX 4 format) around the points. Raises ValueError if it is flat or has too many vertices."""
    pts = [tuple(map(float, p)) for p in points]
    tris = convex_triangles(pts)
    # merge nearly flat neighbours like PhysX; if that leaves an inconsistent outline (almost degenerate
    # points), try a tighter tolerance, and in the end keep the triangles as they are (always valid)
    for tol in (0.0007, 0.0002, 0.00005, None):
        try:
            if tol is None:
                polys = [(_norm(_cross(_sub(pts[b], pts[a]), _sub(pts[c], pts[a]))),
                          -_dot(_norm(_cross(_sub(pts[b], pts[a]), _sub(pts[c], pts[a]))), pts[a]), [a, b, c])
                         for a, b, c in tris]
            else:
                polys = _polygons(pts, tris, tol)
            _check_closed(polys)
            break
        except ValueError:
            if tol is None:
                raise
    used = []
    for nn, d, loop in polys:
        for v in loop:
            if v not in used:
                used.append(v)
    if len(used) > MAX_VERTICES:
        raise ValueError(f"{len(used)} hull vertices: at most {MAX_VERTICES} (simplify the collision shape)")
    if len(polys) > 255:
        raise ValueError("more than 255 polygons")
    remap = {v: i for i, v in enumerate(used)}
    verts = [pts[v] for v in used]
    loops = [[remap[v] for v in loop] for nn, d, loop in polys]
    refs, polygons = bytearray(), []
    for (nn, d, loop), lp in zip(polys, loops):
        mi = min(range(len(verts)), key=lambda i: _dot(nn, verts[i]))
        polygons.append(((nn[0], nn[1], nn[2], d), len(refs), len(lp), mi))
        refs += bytes(lp)
    # edges: polygon pairs, in the order of the sorted vertex pairs
    owner = {}
    for pi, lp in enumerate(loops):
        for k in range(len(lp)):
            owner.setdefault(tuple(sorted((lp[k], lp[(k + 1) % len(lp)]))), []).append(pi)
    if any(len(v) != 2 for v in owner.values()):
        raise ValueError("hull is not closed")
    fbe = bytearray()
    for e in sorted(owner):
        fbe += bytes(sorted(owner[e]))
    fbv = bytearray()
    for i in range(len(verts)):
        ps = [pi for pi, lp in enumerate(loops) if i in lp][:3]
        if len(ps) < 3:
            raise ValueError("a hull vertex is on fewer than 3 polygons")
        fbv += bytes(ps)
    vol, com, inertia = _mass_properties(verts, [(p[0], p[1], lp) for p, lp in zip(polys, loops)])
    inertia = tuple(x * inertia_sign if k not in (0, 4, 8) else x for k, x in enumerate(inertia))
    lo = tuple(min(v[k] for v in verts) for k in range(3))
    hi = tuple(max(v[k] for v in verts) for k in range(3))
    radius = min(abs(_dot(p[0], com) + p[1]) for p in polys)
    e = [hi[k] - lo[k] for k in range(3)]
    big = max(range(3), key=lambda k: e[k])
    r = radius / 3 ** 0.5
    ext = [r, r, r]
    others = [k for k in range(3) if k != big]
    lim = []
    for nn, d, loop in polys:
        if abs(nn[big]) > 1e-9:
            dist = -(_dot(nn, com) + d) - r * sum(abs(nn[k]) for k in others)
            lim.append(dist / abs(nn[big]))
    ext[big] = max(r, min(lim)) if lim else r
    return Hull(verts, polygons, bytes(refs), bytes(fbe), bytes(fbv), 0.0, lo + hi, vol, inertia, com, None,
                radius, tuple(ext), len(owner))


def read_phac(d, decode=True):
    head = d[:10]
    count = struct.unpack_from("<I", d, 6)[0]
    ph, pos = PhAc(head), 10
    for _ in range(count):
        n = struct.unpack_from("<I", d, pos)[0]
        name = d[pos + 4:pos + 4 + n]
        pos += 4 + n
        kind = struct.unpack_from("<I", d, pos)[0]
        if kind != 4:
            raise ValueError(f"record type {kind} is not supported (only convex hulls, type 4)")
        matrix = struct.unpack_from("<16f", d, pos + 4)
        zero, one, rid, one2, size = struct.unpack_from("<IHIII", d, pos + 68)
        if (zero, one, one2) != (0, 1, 1):
            raise ValueError("unexpected record fields")
        cooked = d[pos + 86:pos + 86 + size]
        if d[pos + 86 + size:pos + 88 + size] != b"\xcd\xab":
            raise ValueError("record marker missing")
        rec = Record(name, matrix, rid, raw=cooked)
        if decode:
            rec.hull = read_hull(cooked)
        ph.records.append(rec)
        pos += 88 + size
    if d[pos:] not in (b"\xcd\xab", b"\xcd\xab\x00"):
        raise ValueError("unexpected end of PhAc")
    ph.tail = d[pos:]
    return ph


def write_phac(ph):
    out = bytearray(ph.head[:6] + struct.pack("<I", len(ph.records)))
    for r in ph.records:
        cooked = write_hull(r.hull) if r.hull is not None else r.raw
        out += struct.pack("<I", len(r.name)) + r.name + struct.pack("<I", r.kind) + struct.pack("<16f", *r.matrix)
        out += struct.pack("<IHIII", 0, 1, r.id, 1, len(cooked)) + cooked + b"\xcd\xab"
    out += ph.tail
    return bytes(out)
