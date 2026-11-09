# Checks p3d_lib on real files without writing anything: for each .P3D
#   1. chunk tree parsed and written back = same bytes
#   2. geometry buffer split into blocks and rebuilt unchanged = same bytes
#   3. blocks laid out in reverse order: every part still reads the same vertex and index data
#   4. then laid out in part order again (the default): same bytes as the original
#
#   python p3d_roundtrip.py FOLDER_OR_FILE [...]

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also when run with python -I
from p3d_lib import Geometry, parse, serialize  # noqa: E402


def part_data(geo):
    """Bytes of every stream and index list, per part, read through the SuGe offsets."""
    root = geo.root
    gehe = root.find(b"GeHe")
    buf = b"".join(c.data for c in gehe.children if c.id == b"GHda")
    starts = {b["offset"]: len(b["data"]) for b in geo.blocks}
    out = []
    for p in geo.parts:
        nv, streams, ntri, ioff = Geometry.suge_fields(p)
        out.append(tuple(buf[o:o + starts[o]] for _, o in streams) + (buf[ioff:ioff + starts[ioff]],))
    return out


def check(path):
    orig = path.read_bytes()
    root = parse(orig)
    if serialize(root) != orig:
        return "tree differs"
    geo = Geometry(root)
    geo.rebuild()
    if serialize(root) != orig:
        return "geometry rebuild differs"
    before = part_data(Geometry(parse(orig)))
    geo.blocks.reverse()
    geo.rebuild(layout="current")
    moved = serialize(root)
    after = part_data(Geometry(parse(moved)))
    if before != after:
        return "reordered blocks read back differently"
    # 4. laid out again in the order of the parts, as the game files are: back to the original bytes
    geo.rebuild()
    if serialize(root) != orig:
        return "part-order layout differs from the original"
    return None


def main():
    files = []
    for a in sys.argv[1:]:
        p = Path(a)
        files += sorted(p.rglob("*.P3D")) if p.is_dir() else [p]
    t, bad, kinds = time.time(), [], {}
    for f in files:
        try:
            err = check(f)
        except Exception as e:  # report and go on: one odd file must not hide the others
            err = f"{type(e).__name__}: {e}"
        if err:
            bad.append((f, err))
            kinds[err.split(":")[0]] = kinds.get(err.split(":")[0], 0) + 1
    print(f"{len(files)} files, {len(files) - len(bad)} OK, {len(bad)} not OK, {time.time() - t:.0f} s")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {n:5} {k}")
    for f, err in bad[:25]:
        print(f"  {f}: {err}")


if __name__ == "__main__":
    main()
