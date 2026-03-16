import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

class Config:
    """Classe per gestire la configurazione del progetto."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Setting up the configuration.
        
        Args:
            config_path: path of the JSON. If None, looking for config.json into the script's directory
        """
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.json"
        
        self.config_path = Path(config_path)
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
    
    # ===== PATHS properties =====
    @property
    def get_script_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return script_dir

    @property
    def input_folder(self) -> str:
        """Input folder."""
        return self._data["paths"]["inputFolder"]
    
    @property
    def mesh_name(self) -> str:
        """Mesh name."""
        return self._data["paths"]["meshName"]
    
    @property
    def point_name(self) -> str:
        """Name of the file containing hte points."""
        return self._data["paths"]["pointName"]
    
    @property
    def output_folder(self) -> str:
        """Output folder."""
        return self._data["paths"]["outputFolder"]
    
    # ===== Full path =====
    @property
    def mesh_path(self) -> Path:
        """Mesh full path."""
        script_dir = Path(__file__).parent
        return script_dir / self.input_folder / self.mesh_name
    
    @property
    def point_path(self) -> Path:
        """Point'sfile Mesh full path."""
        script_dir = Path(__file__).parent
        return script_dir / self.input_folder / self.point_name
    
    @property
    def output_path(self) -> Path:
        """Output folder Mesh full path."""
        script_dir = Path(__file__).parent
        return script_dir / self.output_folder
    
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
        """distance of reserach."""
        return self._data["parameters"]["radius"]
    
    @property
    def cores(self) -> int:
        """Number of cores for parallelization."""
        return self._data["parameters"]["cores"]
    
    # ===== GRID properties =====
    @property
    def normal_grid(self) -> np.ndarray:
        """transform the normal to the griglia as array NumPy."""
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
        print(f"✓ Cartella output creata/verificata: {self.output_path}")
    
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
        
        print("✓ All input files exist")
        return True
    
    def print_config(self):
        """Print the current configuration."""
        print("\n" + "="*50)
        print("CONFIGURATION LOADED")
        print("="*50)
        print(f"► Input folder:  {self.input_folder}")
        print(f"► Mesh:          {self.mesh_name}")
        print(f"► Punti:         {self.point_name}")
        print(f"◄ Output folder: {self.output_folder}")
        print(f"\n☼  Tolerance:    {self.tolerance}")
        print(f"☼ Buffer:        {self.buffer}")
        print(f"☼ Radius:        {self.radius}")
        print(f"☼ Cores:         {self.cores}")
        print(f"☼ Normal grid:   {self.normal_grid}")
        print("="*50 + "\n")

    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"