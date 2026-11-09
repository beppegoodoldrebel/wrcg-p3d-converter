# P3D -> OBJ (+ MTL) converter for WRC Generations models (KT Engine, "SCENE___").
#
#   python p3d_to_obj.py MODEL.P3D [OUTPUT_FOLDER] [--all-lods] [--texture BODY_DIFF.dds]
#
#   --all-lods   also export the lower levels of detail (nodes named *_LOD1, *_LOD2, ...)
#   --collision  also export the physics hulls (PhAc) as objects Convex_..., material COLLISION
#   --texture    texture linked (map_Kd) to the body materials (BODY*) in the .mtl file;
#                the path is written relative to the output folder
#
# P3D layout (RIFF, little-endian, "PA__" container):
#   Head  "SCENE___"
#   GeHe  -> GSiz (total size) + 1 MB "GHda" chunks = one buffer with all vertex and index data
#   MeLi  -> for each mesh resource: MRsc (bounding box and sphere) + one GeLi per part (= material):
#            Geom  vertex format, e.g. "COLOTANBNORMPOS_UV0_UV1_"
#            SuGe  vertex count, streams (name + offset in the buffer), triangle count, index offset
#   Nods  -> node tree: NoHe (name, row-major 4x4 matrix, translation at 12..14) + MeRe (resource index)
# Streams (not interleaved): vpos 3 floats · vnor 4 halfs · vuvs UV0 int16/4096 (+ UV1/UV2)
# · vcol 4 bytes · vtan 4 bytes · vski 8 bytes (weights). 16-bit indices, triangle lists.
# Part materials are in the .PSI with the same name: (node, .PMA file) pairs in part order.
# The game is Z-up with the car facing -Y; the OBJ is written Y-up (x, z, -y), as Blender expects.

import os
import re
import struct
import sys
from pathlib import Path


def chunks(b, start, end):
    pos = start
    while pos + 8 <= end:
        cid = b[pos:pos + 4]
        size = struct.unpack_from("<I", b, pos + 4)[0]
        kind = b[pos + 8:pos + 12] if cid in (b"RIFF", b"LIST") else None
        yield cid, kind, (pos + 12 if kind else pos + 8), pos + 8 + size
        pos += 8 + size + (size & 1)


def read_p3d(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"PA__", "not a P3D file"
    top = list(chunks(b, 12, len(b)))
    head = next(b[s:e] for c, k, s, e in top if c == b"Head")
    assert head[:8] == b"SCENE___", "unknown P3D header"

    gehe = next((s, e) for c, k, s, e in top if k == b"GeHe")
    gsiz = next(b[s:e] for c, k, s, e in chunks(b, *gehe) if c == b"GSiz")
    buf = b"".join(b[s:e] for c, k, s, e in chunks(b, *gehe) if c == b"GHda")
    assert len(buf) == struct.unpack_from("<I", gsiz, 2)[0], "unexpected buffer size"

    resources = []
    meli = next((s, e) for c, k, s, e in top if k == b"MeLi")
    for c, k, s, e in chunks(b, *meli):
        if c == b"MRsc":
            resources.append([])
        elif k == b"GeLi":
            part = {}
            for c2, k2, s2, e2 in chunks(b, s, e):
                if c2 == b"Geom":
                    part["format"] = b[s2 + 10:e2 - 2].decode()
                elif k2 == b"SGLi":
                    s3 = next(ss for cc, kk, ss, ee in chunks(b, s2, e2) if cc == b"SuGe")
                    nv, ns = struct.unpack_from("<II", b, s3 + 2)
                    p, streams = s3 + 10, {}
                    for _ in range(ns):
                        streams[b[p:p + 4].decode()] = struct.unpack_from("<I", b, p + 4)[0]
                        p += 8
                    ntri, ioff = struct.unpack_from("<II", b, p)
                    part.update(nverts=nv, streams=streams, ntris=ntri, index_offset=ioff)
            resources[-1].append(part)

    def nodes(s, e):
        out, cur = [], None
        for c, k, ss, ee in chunks(b, s, e):
            if c == b"NoHe":
                n = struct.unpack_from("<I", b, ss + 2)[0]
                cur = {"name": b[ss + 6:ss + 6 + n].decode("latin-1"),
                       "matrix": struct.unpack_from("<16f", b, ss + 6 + n + 2), "mesh": None, "children": []}
                out.append(cur)
            elif c == b"MeRe":
                cur["mesh"] = struct.unpack_from("<I", b, ss)[0]
            elif k == b"Nods":
                cur["children"] += nodes(ss, ee)
        return out

    nods = next((s, e) for c, k, s, e in top if k == b"Nods")
    return buf, resources, nodes(*nods)


def read_psi_materials(path):
    # (node name, .PMA file) pairs in part order: strings prefixed by their length (u32)
    if not Path(path).exists():
        return {}
    b = Path(path).read_bytes()
    strings, i = [], 0
    while i < len(b) - 4:
        n = struct.unpack_from("<I", b, i)[0]
        if 1 <= n <= 200 and i + 4 + n <= len(b) and all(32 <= x < 127 for x in b[i + 4:i + 4 + n]):
            strings.append(b[i + 4:i + 4 + n].decode())
            i += 4 + n
        else:
            i += 1
    mats = {}
    for a, c in zip(strings, strings[1:]):
        if not a.startswith("DATA:/") and c.upper().endswith(".PMA"):
            mats.setdefault(a, []).append(Path(c).stem.upper())
    return mats


def matmul(a, b):
    # row-major 4x4 matrices, row-vector convention (v' = v * M)
    return tuple(sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4))


def transform(m, x, y, z, w=1.0):
    return tuple(x * m[c] + y * m[4 + c] + z * m[8 + c] + w * m[12 + c] for c in range(3))


LOD = re.compile(r"_LOD\d+$", re.I)


def collision_lines(p3d, base):
    """OBJ lines of the physics hulls (PhAc), one object per hull named as in the model, material COLLISION.
    Hull vertices are in model space; polygons are written as they are (Blender splits them as needed)."""
    from p3d_lib import load
    from p3d_phac import read_phac
    ph = next((c for c in load(p3d).walk() if c.id == b"PhAc"), None)
    if ph is None:
        return [], 0
    out, count = [], 0
    for r in read_phac(ph.data).records:
        h = r.hull
        out.append(f"o {r.name.decode('latin-1')}\ng {r.name.decode('latin-1')}_COLLISION\nusemtl COLLISION\n")
        out.extend(f"v {x:.6f} {z:.6f} {-y:.6f}\n" for x, y, z in (transform(r.matrix, *v) for v in h.vertices))
        for plane, vref, n, mi in h.polygons:
            out.append("f " + " ".join(str(base + i) for i in h.vertex_refs[vref:vref + n]) + "\n")
        base += len(h.vertices)
        count += 1
    return out, count


def export(p3d, out_dir=None, all_lods=False, texture=None, collision=False, psi=None):
    """psi: the model's .PSI (material names); default: one folder up from the model, as in the game."""
    p3d = Path(p3d)
    out_dir = Path(out_dir or p3d.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf, resources, roots = read_p3d(p3d)
    psi_path = Path(psi) if psi else p3d.parent.parent / (p3d.stem + ".PSI")
    mats = read_psi_materials(psi_path)
    try:     # per node, also when two nodes share a name (the k-th node with a mesh = the k-th run in the .PSI)
        from p3d_psi import Psi
        psi = Psi(psi_path) if psi_path.exists() else None
    except Exception:
        psi = None
    seen = {}
    obj = [f"# {p3d.name}\nmtllib {p3d.stem}.mtl\n"]
    used_mats = set()
    stats = dict(objects=0, verts=0, tris=0, uv_out=0, flipped=0)
    base = 1

    def walk(node, parent_m):
        nonlocal base
        world = matmul(node["matrix"], parent_m)
        skip = (not all_lods and LOD.search(node["name"])) or node["name"].startswith("Convex_")
        occ = None
        if node["mesh"] is not None:
            occ = seen.get(node["name"], 0)
            seen[node["name"]] = occ + 1
        if node["mesh"] is not None and not skip:
            mlist = mats.get(node["name"], [])
            if psi is not None:
                own = [Path(p.replace("DATA:/", "")).stem.upper() for p, f in psi.materials(node["name"], occ)]
                if len(own) == len(resources[node["mesh"]]):
                    mlist = own
            obj.append(f"o {node['name']}\n")
            stats["objects"] += 1
            for pi, part in enumerate(resources[node["mesh"]]):
                st, nv = part["streams"], part["nverts"]
                pos = [transform(world, *struct.unpack_from("<3f", buf, st["vpos"] + i * 12)) for i in range(nv)]
                nor = [transform(world, *struct.unpack_from("<3e", buf, st["vnor"] + i * 8), w=0.0) for i in range(nv)]
                uvstride = (len(re.findall(r"UV\d_", part["format"])) or 1) * 4
                uvs = []
                for i in range(nv):
                    u, v = struct.unpack_from("<2h", buf, st["vuvs"] + i * uvstride)
                    u, v = u / 4096.0, v / 4096.0
                    stats["uv_out"] += not (-0.01 <= u <= 1.01 and -0.01 <= v <= 1.01)
                    uvs.append((u, 1.0 - v))  # OBJ: v starts at the bottom
                idx = struct.unpack_from(f"<{part['ntris'] * 3}H", buf, part["index_offset"])
                # winding check: face normal vs vertex normals
                agree = 0
                for t in range(0, min(len(idx), 3000), 3):
                    a, b2, c = (pos[idx[t]], pos[idx[t + 1]], pos[idx[t + 2]])
                    e1 = [b2[k] - a[k] for k in range(3)]
                    e2 = [c[k] - a[k] for k in range(3)]
                    fn = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
                    vn = [nor[idx[t]][k] + nor[idx[t + 1]][k] + nor[idx[t + 2]][k] for k in range(3)]
                    agree += 1 if sum(fn[k] * vn[k] for k in range(3)) >= 0 else -1
                flip = agree < 0
                stats["flipped"] += flip
                mat = mlist[pi] if pi < len(mlist) else f"PART_{node['mesh']}_{pi}"
                used_mats.add(mat)
                obj.append(f"g {node['name']}_{mat}\nusemtl {mat}\n")
                obj.extend(f"v {x:.6f} {z:.6f} {-y:.6f}\n" for x, y, z in pos)
                obj.extend(f"vt {u:.6f} {v:.6f}\n" for u, v in uvs)
                obj.extend(f"vn {x:.5f} {z:.5f} {-y:.5f}\n" for x, y, z in nor)
                for t in range(0, len(idx), 3):
                    a, b2, c = idx[t] + base, idx[t + 1] + base, idx[t + 2] + base
                    if flip:
                        b2, c = c, b2
                    obj.append(f"f {a}/{a}/{a} {b2}/{b2}/{b2} {c}/{c}/{c}\n")
                base += nv
                stats["verts"] += nv
                stats["tris"] += part["ntris"]
        for ch in node["children"]:
            walk(ch, world)

    identity = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    for r in roots:
        walk(r, identity)
    hulls = 0
    if collision:
        lines, hulls = collision_lines(p3d, base)
        obj.extend(lines)
        if hulls:
            used_mats.add("COLLISION")

    tex = os.path.relpath(Path(texture).resolve(), out_dir.resolve()).replace("\\", "/") if texture else None
    mtl = []
    for m in sorted(used_mats):
        mtl.append(f"newmtl {m}\nKd 0.8 0.8 0.8\n")
        if tex and m.startswith("BODY"):
            mtl.append(f"map_Kd {tex}\n")
        mtl.append("\n")
    (out_dir / f"{p3d.stem}.obj").write_text("".join(obj), encoding="utf-8")
    (out_dir / f"{p3d.stem}.mtl").write_text("".join(mtl), encoding="utf-8")
    print(f"{p3d.name} -> {out_dir / (p3d.stem + '.obj')}: {stats['objects']} objects, {stats['verts']} vertices, "
          f"{stats['tris']} triangles, {len(used_mats)} materials, UVs outside 0..1: {stats['uv_out']}, "
          f"parts with flipped winding: {stats['flipped']}" + (f", collision hulls: {hulls}" if collision else ""))


USAGE = "usage: p3d_to_obj.py MODEL.P3D [OUTPUT_FOLDER] [--all-lods] [--collision] [--texture BODY_DIFF.dds]"

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # p3d_lib / p3d_phac, also with python -I
    args, texture, all_lods, collision = [], None, False, False
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--all-lods":
            all_lods = True
        elif a == "--collision":
            collision = True
        elif a == "--texture":
            texture = next(it, None)
        else:
            args.append(a)
    if not args or (texture is None and "--texture" in sys.argv):
        raise SystemExit(USAGE)
    export(args[0], args[1] if len(args) > 1 else None, all_lods, texture, collision)
