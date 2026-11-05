# ── Configure here you path  ─────────────────────────────
from pathlib import Path
from grid_to_mesh.project_grid import project_grid

# ── CAMBIA QUI I TUOI PERCORSI ─────────────────────────────
CONFIG_FILE = Path("data/config.json")
MESH_FILE   = Path("data/input/mesh.ply")
GRID_FILE = Path("data/input/36PT.xyz")
OUTPUT_DIR  = Path("output")
OUTPUT_NAME = "projected_grid"
# ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    
        project_grid(
                config=CONFIG_FILE,
                mesh_file=MESH_FILE,
                grid_file=GRID_FILE,
                output_dir=OUTPUT_DIR,
                output_name=OUTPUT_NAME,
        )

# ───────────────────────────────────────────────────────────