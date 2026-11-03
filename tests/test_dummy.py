from pathlib import Path
from g2m_projector.project_grid import project_grid

TESTS = Path(__file__).resolve().parent


def test_project_grid():
    out_file = TESTS / "output" / "projected_grid.xyz"
    out_file.unlink(missing_ok=True)

    project_grid(
        config=TESTS / "input" / "config.json",
        mesh_file=TESTS / "input" / "mesh.ply",
        points_file=TESTS / "input" / "36PT.xyz",
        output_dir=TESTS / "output",
        output_name="projected_grid",
    )

    assert out_file.exists()