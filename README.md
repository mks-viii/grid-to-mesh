# grid-to-mesh

This tool projects a regular grid of points onto mesh surfaces. It works with large and small triangles, as well as flat and almost vertical ones.

The mesh format is `.ply` and the grid is provided in a `.xyz` file.

The algorithm identifies the triangles intersected by the normal to the grid and projects each point along the nadir direction.

It works with a single core or multiple cores: specify the number of cores in the config file.

## Installation

```bash
git clone <repository-url>
cd grid-to-mesh
pip install -e .
```

Requirements: Python `<version>` or later. Main dependencies: `numpy`, `<other libraries>`.

## Usage

1. Put your mesh (`.ply`) and your grid (`.xyz`) in the input folder.
2. Edit `config.json` with the file names and the parameters (see [Configuration](#configuration)).
3. Run the tool:

```bash
python -m grid_to_mesh.main
```

The projected points are saved in the output folder, with the suffix `_proj` added to the file name (e.g. `grid.xyz` becomes `grid_proj.xyz`).

## Configuration

The tool computes the bounding box of the mesh and searches for the triangles contained in it. The parameters you can adjust in `config.json` are:

- **cores**: number of cores used for parallel processing.
- **buffer**: a buffer around the bounding box, to make sure triangles close to the mesh border are included.
- **radius**: maximum search distance for finding the nearest triangles to a point. Set the radius according to the distance between the point grid and the mesh surface.
- **tolerance**: `4e-1` for a high-resolution mesh, `10` for a coarse mesh. If a point has been projected onto the opposite side, increase the tolerance.
- **closed_surface**: set to `true` for closed meshes (e.g. spheres, cylinders), which is very slow. Set to `false` for flat or smoothly curved surfaces, which is fast.

The following image describes the relationship among the config parameters.

![Config](img/buffer_radius_tolerance.png)

### Decimation settings

> **Note:** the decimation tool is currently under development.

The following settings are used by the decimation tool. For the projection you can ignore them and keep the default values:

- **preserveboundary** (boolean): adjust for small circular meshes, e.g. a cylinder.
- **preservenormal** (boolean)
- **preservetopology** (boolean)

## Example

A test dataset is available in the `tests/` folder. To run the tests:

```bash
pytest -ls
```

In the following image, the points of a regular grid (yellow) are projected onto the mesh along the normal direction (yellow line). The projected points are shown as red dots.

*Note: the image is for visualization purposes only and was produced using external software.*


Projected points:

![Output](img/36PT_proj.png)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.