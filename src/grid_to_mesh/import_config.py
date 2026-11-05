import json
import numpy as np
from pathlib import Path


class Config:
    """Class for setting up the configuration."""

    def __init__(self, config_path: Path, mesh_path: Path,
                 point_path: Path, output_path: Path):
        """
        Setting up the configuration.

        Args:
            config_path: JSON file with the parameters
            mesh_path:   mesh file (.ply)
            point_path:  points file (.xyz)
            output_path: output folder (created if missing)
        """
        self.config_path = Path(config_path).resolve()
        self.mesh_path = Path(mesh_path).resolve()
        self.point_path = Path(point_path).resolve()
        self.output_path = Path(output_path).resolve()
        self._load_config()

    def _load_config(self):
        """Setting up the JSON configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Sorry, I cannot find the configuration file: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Parsing error in the JSON: {e}")

    # ===== Parameters properties =====
    @property
    def tolerance(self) -> float:
        """Tolerance around the point."""
        return self._data["parameters"]["tolerance"]

    @property
    def buffer(self) -> float:
        """Buffer."""
        return self._data["parameters"]["buffer"]

    @property
    def radius(self) -> float:
        """Distance of research."""
        return self._data["parameters"]["radius"]

    @property
    def closed_surface(self) -> bool:
        """
        True  → sphere, cylinder or closed surfaces (slow, 3D search).
        False → flat, smooth curved surfaces like terrain (fast, 2D search).
        """
        return self._data["parameters"].get("closed_surface", False)

    @property
    def cores(self) -> int:
        """Number of cores for parallelization."""
        return self._data["parameters"]["cores"]

    # ===== GRID properties =====
    @property
    def normal_grid(self) -> np.ndarray:
        """Transform the normal to the grid as NumPy array."""
        return np.array(self._data["grid"]["normalGrid"])

    # ===== Decimator properties =====
    @property
    def preserveboundary(self) -> bool:
        return self._data["decimator"]["preserveboundary"]

    @property
    def preservenormal(self) -> bool:
        return self._data["decimator"]["preservenormal"]

    @property
    def preservetopology(self) -> bool:
        return self._data["decimator"]["preservetopology"]

    # =====  UTILITY =====
    def create_output_folder(self):
        """Create the output folder if it doesn't exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output folder created/verified: {self.output_path}")

    def validate_paths(self) -> bool:
        """Verify that input files exist."""
        errors = []

        if not self.mesh_path.exists():
            errors.append(f"X Mesh not found: {self.mesh_path}")

        if not self.point_path.exists():
            errors.append(f"X Point's file not found: {self.point_path}")

        if errors:
            for error in errors:
                print(error)
            return False

        print("All input files exist!")
        return True

    def print_config(self):
        """Print the current configuration."""
        print("\n" + "=" * 50)
        print("CONFIGURATION LOADED")
        print("=" * 50)
        print(f"► Mesh:          {self.mesh_path}")
        print(f"► Points:        {self.point_path}")
        print(f"◄ Output folder: {self.output_path}")
        print(f"\n☼ Tolerance:     {self.tolerance}")
        print(f"☼ Buffer:        {self.buffer}")
        print(f"☼ Radius:        {self.radius}")
        print(f"☼ Cores:         {self.cores}")
        print(f"☼ Normal grid:   {self.normal_grid}")
        print(f"☼ Closed surface: {self.closed_surface}")
        print("=" * 50 + "\n")

    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"