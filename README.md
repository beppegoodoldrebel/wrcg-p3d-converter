# WRCG P3D Converter

Export **WRC Generations** 3D models (`.P3D`) to **OBJ + MTL**, ready to import in Blender.

- Every car part becomes its own object, placed with the model's node transforms (hood, doors, bumpers… in place).
- One material per part, named after the game material (`BODY`, `GLASS`, `WINDSCREEN`, `FRONTOPTICS`, `COCKPIT`…).
- Normals and livery UVs included; the livery texture can be linked to the body materials.
- Parts edited in Blender go back into the model (`obj_to_p3d.py`): new or removed parts, physics hulls, with a window to do it all (`p3d_converter.bat`).
- Pure Python 3, no extra packages (the window uses tkinter, included with Python on Windows).

> Unofficial fan tool, not affiliated with KT Racing or Nacon. Parts edited in Blender can be put back into the model with `obj_to_p3d.py` (see below); whole new cars are not supported yet.

## Requirements

- **Python 3.8+**
- The game's loose files: run `WRCGUnpackager.exe` from [WRCGModTools](https://github.com/Avery3R/WRCGModTools) by Avery3R, then set `Engine.UsePkg = false;` in `Common\Settings\SETTINGS.CFG`, or the game will not read the edited files. Models are in `WIN32\GRAPH\CARS\...\<CAR>\MESH\*.P3D`.

## Window (GUI)

Double-click `p3d_converter.bat` (or `p3d_gui.pyw`). It finds the game's `WIN32\GRAPH\CARS` folder in the usual Steam locations.

- **P3D → OBJ**: choose a model and an output folder (default `Desktop\WRCG_MODELS\<MODEL>`). Optional: lower levels of detail, physics hulls (on by default), a body texture for Blender.
- **OBJ → P3D**: choose the game model and the OBJ edited in Blender, then **Build P3D**. The rebuilt model goes to `REBUILT\MESH\` next to the OBJ, with its `.PSI` in `REBUILT\` when parts were added or removed. Options: same shape in the lower levels of detail, rewrite unchanged parts, only some nodes.
- **Wipers ↔ Blender**: exports a car's wipers (meshes with weights, bones, animation, plus reference bodywork) as a glTF, and rebuilds the model and `WIPERS.PAN` from the file edited in Blender. See [Wipers in Blender](#wipers-in-blender-p3d_wiperspy).
- **Install in game** copies the rebuilt model (and `.PSI`) over the game's files. The first time, it keeps the game's files in `ORIGINALI` folders: `MESH\ORIGINALI\` for the model, `<CAR>\ORIGINALI\` for the `.PSI`. This is the same convention as the WRCG PTX Converter. **Restore originals** puts them back. Both need the game closed.
- Once a backup exists, export and build always start from it, so changes do not pile up on an already modified model. Install for the first time with the original model in the game: the first backup holds whatever is in the game at that moment.
- Messages and errors appear in the log at the bottom.

## Usage

```
python p3d_to_obj.py MODEL.P3D [OUTPUT_FOLDER] [--all-lods] [--texture BODY_DIFF.dds]
```

- `--all-lods` – also export the lower levels of detail (`*_LOD1`, `*_LOD2`…). By default only the full-detail model is exported.
- `--collision` – also export the physics hulls as objects `Convex_…` with material `COLLISION`, in model space. Edit them in Blender and `obj_to_p3d.py` rebuilds them; see below.
- `--texture` – links a texture (`map_Kd`) to the body materials. Car textures are `.PTX` files: convert them to DDS first with the **WRCG PTX Converter** (e.g. `<CAR>\<LIVERY>\BODY_DIFF.PTX`). Blender reads DDS directly.

Example:

```
python p3d_to_obj.py "...\LEGENDS\ALPINE_A110\MESH\RENAULT_ALPINE_A110.P3D" Alpine --texture Alpine\BODY_DIFF.dds
```

In Blender: **File → Import → Wavefront (.obj)** with default settings.

Wheels are separate files (`RIM_TARMAC.P3D`, `RIM_GRAVEL.P3D`, `RIM_SNOW.P3D`) and can be exported the same way.

## Putting edited parts back (`obj_to_p3d.py`)

```
python obj_to_p3d.py ORIGINAL.P3D EDITED.obj OUTPUT.P3D [--only Hood,Spoiler] [--same-lod] [--psi MODEL.PSI]
```

1. Export the model with `p3d_to_obj.py`, import the OBJ in Blender (default settings), edit the parts you want.
2. In Blender use **File → Export → Wavefront (.obj)**. Keep the default axes (Y up, −Z forward), with UV coordinates, normals and materials. Triangulating is optional.
3. Run `obj_to_p3d.py` with the original P3D. It writes a new file and never changes the original.

How it works:

- Objects are matched to the model's nodes by name. A second node with the same name is `Name.001`, as Blender calls it.
- Faces are grouped by material name (`BODY`, `GLASS`…). The names come from the model's `.PSI`, found the same way as the exporter does, or given with `--psi`.
  - A material the node already has replaces that material's parts.
  - A material the node does not have, but the car uses somewhere else, becomes a **new part**. A material that no face uses any more has its parts **removed**.
  - More than 65535 vertices for one material are **split** into several parts, because the indices are 16-bit.
  - When parts are added or removed, the `.PSI` (`p3d_psi.py`, materials per part) is written as well: one folder up from `OUTPUT.P3D` as in the game, or to `--psi-out`. It is never written over the original. A material the car does not use at all needs its own `.PMA` file, which is not supported yet.
- Only nodes whose shape changed are rewritten, unless `--only` lists them. The tolerance is 0.2 mm, because the OBJ keeps 6 decimals.
- Positions, normals and UV0 come from the OBJ. The vertex colour, the tangents, the 4th normal value and UV1/UV2 cannot travel in an OBJ. They are copied from the nearest vertex (k-d tree) of the node's original parts with that material, or, for a new part, of the parts with that material elsewhere in the car. The triangle order follows those parts. Bounding boxes are recomputed.
- `--same-lod` also puts the new shape into `Name_LOD1`, `Name_LOD2`… when they have the same parts. Otherwise the lower levels of detail keep the old shape, which is seen from far away.
- Limits: skinned parts cannot be rewritten here (wipers have their own tool, see below). New nodes (objects whose name is not in the model) are skipped.
- Physics hulls: an object `Convex_…` replaces the hull with the same name. The new hull is the convex hull of the object's vertices, built and cooked by `p3d_phac.py`. Hulls can be replaced but not added, and each one can have at most 32 vertices (above that PhysX needs extra data that is not written yet).

## Wipers in Blender (`p3d_wipers.py`)

```
python p3d_wipers.py export MODEL.P3D WIPERS.PAN OUT.gltf
python p3d_wipers.py import MODEL.P3D WIPERS.PAN EDITED.glb OUT_FOLDER
```

The glTF holds a real armature (`Ref_Wipers` and the `D_Wiper_*` bones), the wiper meshes with their weights, UVs and normals, the animation on the bones, and `REF_*` bodywork objects that are never read back. Tested with Blender 2.80: importing and exporting unchanged comes back within 0.001 mm and 0.007°.

In Blender, import with File > Import > glTF 2.0. You can edit the meshes, weights, bone positions and keyframes. Export with File > Export > glTF 2.0 with Animation on, keeping the object and bone names.

- The animation starts from the first keyframe of the file, and that pose is the wipers' rest pose in the game.
- The game plays wipers on an additive layer, so only the motion from the first key counts.
- `.PAN` files have rotation keys only: a bone can rotate, but it cannot change position relative to its parent during the animation. Translation keys are reported in the log and ignored. To move a bone, move it in Edit Mode.
- A wiper can only use the bones listed in its skin (`D_Wiper_L_00/01/02/04`, `D_Wiper_R_00/01/02` on the cars checked). Weights on other bones stop the build with a message.
- Each bone keeps its rotation axes from the game; its position and motion come from Blender.

## Physics hulls (`p3d_phac.py`)

The `PhAc` chunk holds one record per hull: name, matrix, a 32-bit id, and a PhysX 4 cooked convex mesh (serial version 13). The record layout and the cooked format are written at the top of `p3d_phac.py`.

- `read_phac()` / `write_phac()` give back the same bytes for all 177 hulls of the 36 car models in the game, including the 71 with big convex data.
- `build_hull(points)` makes a new hull: convex hull, coplanar faces merged into polygons, adjacency, mass properties and internal radius/extents. It follows the rules found in the game's hulls. Rebuilt from their own vertices, the 106 small car hulls of the game give the same centre of mass in 101 cases, inertia in 100 and internal extents in 105 (within 0.2–0.5 %).
- In the game, equal cooked data always have the same id; the formula is not known. New hulls get the CRC32 of their cooked data.

Tested in the game: with the bottom of all 7 Escort hulls moved below the tyres, built with `build_hull()` and given CRC32 ids, the game loads the model and the car rides on the invisible blocks, lifted off the ground. So the game accepts these cooked hulls and these ids.

Checked without the game (scripts kept outside this repository):

- An OBJ that was not edited gives back an identical file.
- P3D → OBJ → P3D (every node forced) → OBJ gives identical triangles and UVs for all 111 parts of the Escort.
- A spoiler moved in the OBJ only rewrites the spoiler.
- Faces with material `GLASS` added to the hood give a third hood part, and the `.PSI` lists `BODY`, `BODY_NRM`, `GLASS`. With the hood's `BODY_NRM` faces deleted, that part is removed. A 70,756-vertex spoiler is split into two `BODY` parts of 65,535 and 5,488 vertices.
- All 277 car `.PSI` files of the game are written back byte for byte by `p3d_psi.py`.

Tested in the game: an Escort hood reshaped in Blender 2.80 (`Hood` and `Hood_LOD1` rewritten with `--same-lod`) loads and shows the new shape in the garage. New parts work too. Three copies of the hood all show, without errors: a new `ENGINE` part on `Hood`, faces added to `Hood`'s existing `BODY` part, and a new `ENGINE` part on `Exhaust`.

## Writing P3D files (`p3d_lib.py`)

A library for tools that change models; it has no command of its own yet.

- `parse()` / `serialize()` read the whole chunk tree, known chunks or not, and write it back. An unchanged file comes out byte for byte.
- `Geometry(tree)` splits the shared vertex and index buffer into blocks: one per stream of each part, plus its index list. `rebuild()` lays them out again: vertex streams start at multiples of 16 bytes, index lists at multiples of 4. It also updates the offsets in `SuGe`, the size in `GSiz` and the 1 MB `GHda` chunks. Blocks can be replaced with data of another length.

Check on real files, writing nothing:

```
python p3d_roundtrip.py "...\WIN32\GRAPH"
```

For each file it checks three things: the tree written back is identical; the geometry rebuilt without changes is identical; with the blocks laid out in reverse order, every part still reads the same data. All 2995 P3D files of the game's `WIN32\GRAPH` pass (game version 1.5.26.0). Tested in the game as well: an Escort model with a resized rear spoiler, written with this library, loads and shows correctly. In that model the part's `MRsc` bounding box and sphere were recomputed: floats min xyz, max xyz, 0, centre xyz, radius (half the box diagonal).

Not handled yet: bones (`Skin`, `RTbl`) and the encoding of the 4-byte `vtan` stream. They are kept as they are. The physics hulls (`PhAc`) are handled by `p3d_phac.py`.

## P3D format (summary)

RIFF container (`PA__`, little-endian), header `SCENE___`:

| Chunk | Content |
|---|---|
| `GeHe` | `GSiz` (u16 version, u32 buffer size, u32 chunk size = 1 MB, u32 resource count) + `GHda` chunks: one buffer with all vertex and index data |
| `MeLi` | mesh resources: `MRsc` (tag `MESH`/`SKINMESH`, bounding box and sphere, **number of parts**) + one `GeLi` per part: `Geom` (vertex format, e.g. `COLOTANBNORMPOS_UV0_UV1_`) and `SuGe` (vertex count, streams with buffer offsets, triangle count, index offset) |

Rules seen in every game file, which a writer must keep: all parts of a resource have the same vertex format; the part count at the end of `MRsc` matches the `GeLi` that follow it, since the game reads the number of geometries from there; the buffer holds the parts in `MeLi` order, each with its streams and then its index list. A model that broke one of these loaded with errors, hung or crashed, or did not show the new part.
| `Nods` | node tree: `NoHe` (name, row-major 4×4 matrix) + `MeRe` (mesh resource index) |

Streams are not interleaved: `vpos` 3 floats · `vnor` 4 halfs · `vuvs` UV0 as int16/4096 (+ UV1/UV2) · `vcol` 4 bytes · `vtan` 4 bytes · `vski` 8 bytes (skin weights). Indices are 16-bit triangle lists. Part materials come from the model's `.PSI` file (node → `.PMA` pairs in part order). The game is Z-up with the car facing −Y; the OBJ is written Y-up.

Not exported yet: vertex colors, tangents, UV1/UV2, skin weights.

## Credits

- [Avery3R](https://github.com/Avery3R) – [WRCGModTools](https://github.com/Avery3R/WRCGModTools) (unpacker / enabling mods).

## License

[MIT](LICENSE). WRC Generations and its assets belong to their respective owners; this repository contains no game files.
