# ── Configure here you path  ─────────────────────────────
from pathlib import Path
from g2m_projector.project_grid import project_grid

# ── CAMBIA QUI I TUOI PERCORSI ─────────────────────────────
CONFIG_FILE = Path("examples/data/config.json")
MESH_FILE   = Path("examples/data/input/mesh.ply")
POINTS_FILE = Path("examples/data/input/36PT.xyz")
OUTPUT_DIR  = Path("examples/output")
OUTPUT_NAME = "projected_grid"
# ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_grid(
        config=CONFIG_FILE,
        mesh_file=MESH_FILE,
        points_file=POINTS_FILE,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME,
    )
# ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_grid(
        config=CONFIG_FILE,
        mesh_file=MESH_FILE,
        points_file=POINTS_FILE,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME,
    )