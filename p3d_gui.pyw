# GUI for the WRCG P3D converter: export models to OBJ, put the parts edited in Blender back, install the
# rebuilt model in the game (with a backup) and put the originals back.
# Start it by double-clicking p3d_converter.bat (or this file, if .pyw is associated with Python).

import contextlib
import io
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3d_to_obj  # noqa: E402
import p3d_tools  # noqa: E402
from obj_to_p3d import import_obj  # noqa: E402
from p3d_wipers import export_wipers, import_wipers  # noqa: E402

P3D_TYPES = [("Game models", "*.P3D"), ("All files", "*.*")]
PAN_TYPES = [("Game animations", "*.PAN"), ("All files", "*.*")]
GLTF_TYPES = [("glTF", "*.gltf *.glb"), ("All files", "*.*")]
OBJ_TYPES = [("Wavefront OBJ", "*.obj"), ("All files", "*.*")]
PSI_TYPES = [("Scene instances", "*.PSI"), ("All files", "*.*")]
DDS_TYPES = [("Textures", "*.dds *.png"), ("All files", "*.*")]
WORK = Path.home() / "Desktop" / "WRCG_MODELS"


class FileRow:
    """Label + entry + Browse... button (a file, or a folder when folder=True)."""

    def __init__(self, parent, row, label, types=None, save=False, folder=False, on_change=None, start=None):
        self.var = tk.StringVar()
        self.types, self.save, self.folder, self.on_change, self.start = types, save, folder, on_change, start
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=self.var, width=78).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(parent, text="Browse...", command=self.browse).grid(row=row, column=2, padx=6)

    def browse(self):
        cur = self.get()
        start = (cur if cur and cur.is_dir() else cur.parent if cur else None) or self.start
        kw = dict(initialdir=str(start) if start and Path(start).exists() else None)
        if self.folder:
            path = filedialog.askdirectory(**kw)
        elif self.save:
            path = filedialog.asksaveasfilename(filetypes=self.types, defaultextension=self.types[0][1][1:], **kw)
        else:
            path = filedialog.askopenfilename(filetypes=self.types, **kw)
        if path:
            self.set(Path(path))
            if self.on_change:
                self.on_change(Path(path))

    def get(self):
        v = self.var.get().strip().strip('"')
        return Path(v) if v else None

    def set(self, path):
        self.var.set(str(path) if path else "")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("P3D converter – WRC Generations")
        self.cars = p3d_tools.find_game_cars()
        self.queue = queue.Queue()
        self.busy = False
        self.built = None           # (game model, rebuilt model, rebuilt .PSI or None) of the last build
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=False, padx=8, pady=(8, 4))
        nb.add(self.export_tab(nb), text="  P3D → OBJ  ")
        nb.add(self.import_tab(nb), text="  OBJ → P3D  ")
        nb.add(self.wipers_tab(nb), text="  Wipers ↔ Blender  ")
        box = ttk.LabelFrame(self, text="Log")
        box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(box, height=14, wrap="word", state="disabled")
        sb = ttk.Scrollbar(box, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log(f"Game cars folder: {self.cars}" if self.cars else "Game folder not found: choose the files by hand.")
        self.after(100, self.poll)

    # ------------------------------------------------------------ tabs

    def export_tab(self, nb):
        f = ttk.Frame(nb)
        f.columnconfigure(1, weight=1)
        self.e_model = FileRow(f, 0, "Model (.P3D)", P3D_TYPES, on_change=self.export_model_chosen, start=self.cars)
        self.e_out = FileRow(f, 1, "Output folder", folder=True, start=WORK)
        self.e_tex = FileRow(f, 2, "Body texture (optional)", DDS_TYPES)
        self.e_lods = tk.BooleanVar(value=False)
        self.e_coll = tk.BooleanVar(value=True)
        opts = ttk.Frame(f)
        opts.grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(opts, text="Lower levels of detail (_LOD1, _LOD2...)", variable=self.e_lods).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opts, text="Physics hulls (Convex_...)", variable=self.e_coll).pack(side="left")
        ttk.Label(f, text="Material names come from the model's .PSI (one folder up, as in the game); the saved "
                          "original is used when one exists.", foreground="#555").grid(row=4, column=1, sticky="w", padx=6)
        ttk.Button(f, text="Export to OBJ", command=self.do_export).grid(row=5, column=1, sticky="e", padx=6, pady=8)
        return f

    def import_tab(self, nb):
        f = ttk.Frame(nb)
        f.columnconfigure(1, weight=1)
        self.i_model = FileRow(f, 0, "Game model (.P3D)", P3D_TYPES, on_change=self.import_model_chosen, start=self.cars)
        self.i_src = ttk.Label(f, text="", foreground="#555")
        self.i_src.grid(row=1, column=1, sticky="w", padx=6)
        self.i_obj = FileRow(f, 2, "Edited OBJ", OBJ_TYPES, on_change=self.obj_chosen, start=WORK)
        self.i_out = FileRow(f, 3, "Rebuilt model", P3D_TYPES, save=True, start=WORK)
        self.i_lod = tk.BooleanVar(value=True)
        self.i_force = tk.BooleanVar(value=False)
        self.i_only = tk.StringVar()
        opts = ttk.Frame(f)
        opts.grid(row=4, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(opts, text="Same shape in the lower levels of detail", variable=self.i_lod).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opts, text="Rewrite unchanged parts too", variable=self.i_force).pack(side="left", padx=(0, 16))
        ttk.Label(opts, text="Only nodes:").pack(side="left")
        ttk.Entry(opts, textvariable=self.i_only, width=24).pack(side="left", padx=4)
        btns = ttk.Frame(f)
        btns.grid(row=5, column=1, sticky="e", padx=6, pady=8)
        ttk.Button(btns, text="Build P3D", command=self.do_import).pack(side="left", padx=4)
        self.b_install = ttk.Button(btns, text="Install in game", command=self.do_install, state="disabled")
        self.b_install.pack(side="left", padx=4)
        ttk.Button(btns, text="Restore originals", command=self.do_restore).pack(side="left", padx=4)
        ttk.Label(f, text="Install keeps a copy of the game's files in ORIGINALI folders the first time: do it "
                          "with the original model in the game. Close the game before installing or restoring.",
                  foreground="#555", wraplength=640, justify="left").grid(row=6, column=1, sticky="w", padx=6, pady=(0, 6))
        return f

    def wipers_tab(self, nb):
        f = ttk.Frame(nb)
        f.columnconfigure(1, weight=1)
        self.w_model = FileRow(f, 0, "Model (.P3D)", P3D_TYPES, on_change=self.wipers_model_chosen, start=self.cars)
        self.w_pan = FileRow(f, 1, "Animation (WIPERS.PAN)", PAN_TYPES, start=self.cars)
        self.w_gltf = FileRow(f, 2, "Export to (.gltf)", GLTF_TYPES, save=True, start=WORK)
        self.w_edit = FileRow(f, 3, "Edited in Blender (.glb/.gltf)", GLTF_TYPES, start=WORK)
        self.w_out = FileRow(f, 4, "Output folder", folder=True, start=WORK)
        btns = ttk.Frame(f)
        btns.grid(row=5, column=1, sticky="e", padx=6, pady=8)
        ttk.Button(btns, text="Export for Blender", command=self.do_wipers_export).pack(side="left", padx=4)
        ttk.Button(btns, text="Build from Blender", command=self.do_wipers_import).pack(side="left", padx=4)
        ttk.Label(f, text="In Blender: File > Import > glTF, edit the meshes, the armature and the keyframes, then "
                          "File > Export > glTF 2.0 with Animation on. Keep the object and bone names; REF_* objects are "
                          "only bodywork for reference. Bones can rotate in the animation, not change length.",
                  foreground="#555", wraplength=640, justify="left").grid(row=6, column=1, sticky="w", padx=6, pady=(0, 6))
        return f

    # ------------------------------------------------------------ defaults

    def wipers_model_chosen(self, path):
        pan = path.parent.parent / "ANIM" / "WIPERS.PAN"
        self.w_pan.set(pan if pan.exists() else None)
        self.w_gltf.set(WORK / path.stem / "WIPERS" / (path.stem + "_WIPERS.gltf"))
        self.w_out.set(WORK / path.stem / "WIPERS" / "REBUILT")

    def export_model_chosen(self, path):
        self.e_out.set(WORK / path.stem)

    def import_model_chosen(self, path):
        src, psi = p3d_tools.sources(path)
        saved = src != path
        self.i_src.configure(text=("Built from the saved original " if saved else "Built from ") + str(src)
                             + ("" if psi else "  (no .PSI found: material names PART_..., parts cannot be added)"))
        if not self.i_obj.get():
            obj = WORK / path.stem / (path.stem + ".obj")
            if obj.exists():
                self.obj_chosen(obj)
                self.i_obj.set(obj)
        self.built = None
        self.b_install.configure(state="disabled")

    def obj_chosen(self, path):
        model = self.i_model.get()
        stem = model.stem if model else path.stem
        self.i_out.set(path.parent / "REBUILT" / "MESH" / (stem + ".P3D"))

    # ------------------------------------------------------------ actions

    def run(self, title, work):
        if self.busy:
            return
        self.busy = True
        self.log(f"--- {title}")

        def target():
            out = Writer(self.queue)
            try:
                with contextlib.redirect_stdout(out):
                    work()
            except Exception as e:  # shown in the log and in a message box
                self.queue.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                self.queue.put(("done", None))

        threading.Thread(target=target, daemon=True).start()

    def do_wipers_export(self):
        model, pan, out = self.w_model.get(), self.w_pan.get(), self.w_gltf.get()
        if not model or not model.exists() or not pan or not pan.exists() or not out:
            return messagebox.showwarning("P3D converter", "Choose the model, its WIPERS.PAN and where to export.")

        def work():
            export_wipers(model, pan, out)
            print(f"Open in Blender: File > Import > glTF 2.0 > {out}")
        self.run("Wipers: export", work)

    def do_wipers_import(self):
        model, pan, edit, out = self.w_model.get(), self.w_pan.get(), self.w_edit.get(), self.w_out.get()
        if not all((model, pan, edit, out)) or not edit.exists() or not model.exists() or not pan.exists():
            return messagebox.showwarning("P3D converter", "Choose the model, its WIPERS.PAN, the file from Blender and the output folder.")
        if out.resolve() in (model.parent.resolve(), pan.parent.resolve()):
            return messagebox.showwarning("P3D converter", "The output folder must not be the game's folder.")

        def work():
            p, a, psi = import_wipers(model, pan, edit, out)
            print(f"Rebuilt: {p}\n         {a}\n         {psi}\nCopy them over the game's files with the game closed "
                  "(.P3D to MESH, .PAN to ANIM, .PSI to the car's folder), keeping a backup.")
        self.run("Wipers: build", work)

    def do_export(self):
        model, out = self.e_model.get(), self.e_out.get()
        if not model or not model.exists():
            return messagebox.showwarning("P3D converter", "Choose a model (.P3D).")
        if not out:
            out = WORK / model.stem
            self.e_out.set(out)
        src, psi = p3d_tools.sources(model)
        tex = self.e_tex.get()
        lods, coll = self.e_lods.get(), self.e_coll.get()     # Tk values are read here, not in the worker

        def work():
            if src != model:
                print(f"Using the saved original {src}")
            p3d_to_obj.export(src, out, lods, str(tex) if tex else None, coll, psi)
            print(f"Open in Blender: File > Import > Wavefront (.obj) > {out / (model.stem + '.obj')}")
        self.run("Export", work)

    def do_import(self):
        model, obj, out = self.i_model.get(), self.i_obj.get(), self.i_out.get()
        if not model or not model.exists() or not obj or not obj.exists() or not out:
            return messagebox.showwarning("P3D converter", "Choose the game model, the edited OBJ and the output.")
        src, psi = p3d_tools.sources(model)
        if out.resolve() in (model.resolve(), src.resolve()):
            return messagebox.showwarning("P3D converter", "The rebuilt model must be a new file.")
        out.parent.mkdir(parents=True, exist_ok=True)
        out_psi = out.parent.parent / (model.stem + ".PSI")
        only = {x.strip() for x in self.i_only.get().split(",") if x.strip()} or None
        same_lod, force = self.i_lod.get(), self.i_force.get()     # Tk values are read here, not in the worker

        def work():
            if out_psi.exists():
                out_psi.unlink()      # an old parts list must not be installed with this model
            done = import_obj(src, obj, out, only, same_lod, force, log=print, psi=psi, psi_out=out_psi)
            self.built = (model, out, out_psi if out_psi.exists() else None)
            print(f"Rebuilt nodes: {done or 'none'}." + (f" New parts list: {out_psi}" if out_psi.exists() else ""))
            self.queue.put(("built", None))
        self.run("Build", work)

    def do_install(self):
        if not self.built:
            return
        model, new, psi = self.built
        if not messagebox.askyesno("P3D converter", f"Replace the game's model\n{model}\n" +
                                   (f"and its parts list\n{p3d_tools.psi_of(model)}\n" if psi else "") +
                                   "with the rebuilt one? The originals are kept in ORIGINALI folders."):
            return
        self.run("Install", lambda: p3d_tools.install(model, new, psi, log=print))

    def do_restore(self):
        model = self.i_model.get()
        if not model:
            return messagebox.showwarning("P3D converter", "Choose the game model.")
        if not messagebox.askyesno("P3D converter", f"Put the saved originals of\n{model}\nback in the game?"):
            return
        self.run("Restore", lambda: p3d_tools.restore(model, log=print))

    # ------------------------------------------------------------ log

    def log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip("\n") + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def poll(self):
        try:
            while True:
                kind, text = self.queue.get_nowait()
                if kind == "text":
                    self.log(text)
                elif kind == "error":
                    self.log("ERROR: " + text)
                    messagebox.showerror("P3D converter", text)
                elif kind == "built":
                    self.b_install.configure(state="normal" if self.built and self.cars and
                                             str(self.built[0]).upper().startswith(str(self.cars.parent.parent).upper())
                                             else "disabled")
                elif kind == "done":
                    self.busy = False
        except queue.Empty:
            pass
        self.after(100, self.poll)


class Writer(io.TextIOBase):
    """print() from the worker thread, line by line into the log."""

    def __init__(self, q):
        self.q, self.buf = q, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.q.put(("text", line))
        return len(s)


if __name__ == "__main__":
    App().mainloop()
