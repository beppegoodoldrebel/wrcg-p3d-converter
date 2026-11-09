# Helpers for the GUI: where the original model files are, installing a rebuilt model in the game with a
# backup, and putting the originals back.
#
# Backups follow the WRCG PTX Converter convention: an ORIGINALI folder next to each changed file
#   ...\<CAR>\MESH\ORIGINALI\<MODEL>.P3D       ...\<CAR>\ORIGINALI\<MODEL>.PSI
# They are made the first time a file is replaced, so they hold whatever was in the game at that moment:
# install on an unmodified game (or one with the originals back).

import shutil
import string
import subprocess
from pathlib import Path

GAME_DIR = "WRC Generations - The FIA WRC Official Game"


def find_game_cars():
    """WIN32\\GRAPH\\CARS of the game in the usual Steam folders on every drive, or None."""
    for drive in string.ascii_uppercase:
        for library in (r"SteamLibrary\steamapps\common", r"Program Files (x86)\Steam\steamapps\common",
                        r"Program Files\Steam\steamapps\common", r"Steam\steamapps\common"):
            path = Path(f"{drive}:\\") / library / GAME_DIR / "WIN32" / "GRAPH" / "CARS"
            if path.is_dir():
                return path
    return None


def psi_of(model):
    """The .PSI the game uses with this model: one folder up from MESH, same name."""
    model = Path(model)
    return model.parent.parent / (model.stem + ".PSI")


def backups(model):
    model = Path(model)
    psi = psi_of(model)
    return model.parent / "ORIGINALI" / model.name, psi.parent / "ORIGINALI" / psi.name


def sources(model):
    """(model, .PSI) to build from: the saved originals when they exist, otherwise the files themselves."""
    b_model, b_psi = backups(model)
    psi = psi_of(model)
    return (b_model if b_model.exists() else Path(model)), (b_psi if b_psi.exists() else psi if psi.exists() else None)


def game_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq WRCG.exe", "/NH"], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    return "WRCG.exe" in out


def install(model, new_model, new_psi=None, log=print):
    """Copies new_model over model (and new_psi over its .PSI), saving the originals first. Without new_psi
    the original .PSI is put back if it was saved, since the new model was built with the original parts list."""
    if game_running():
        raise RuntimeError("Close the game first.")
    model, new_model = Path(model), Path(new_model)
    if new_model.resolve() == model.resolve():
        raise RuntimeError("The rebuilt model is the game file itself.")
    b_model, b_psi = backups(model)
    psi = psi_of(model)
    if not b_model.exists():
        b_model.parent.mkdir(exist_ok=True)
        shutil.copy2(model, b_model)
        log(f"Original saved: {b_model}")
    if new_psi and not b_psi.exists() and psi.exists():
        b_psi.parent.mkdir(exist_ok=True)
        shutil.copy2(psi, b_psi)
        log(f"Original saved: {b_psi}")
    shutil.copy2(new_model, model)
    log(f"Installed: {model}")
    if new_psi:
        shutil.copy2(new_psi, psi)
        log(f"Installed: {psi}")
    elif b_psi.exists():
        shutil.copy2(b_psi, psi)
        log(f"Original parts list put back: {psi}")


def restore(model, log=print):
    if game_running():
        raise RuntimeError("Close the game first.")
    b_model, b_psi = backups(model)
    done = False
    for b, target in ((b_model, Path(model)), (b_psi, psi_of(model))):
        if b.exists():
            shutil.copy2(b, target)
            log(f"Original put back: {target}")
            done = True
    if not done:
        log("No saved originals for this model: nothing to do.")
