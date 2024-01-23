from builtins import range

import numpy as np
from numpy import cos, pi, sin
from tvtk.api import tvtk

# from enthought.mayavi.scripts import mayavi2


def aligned_points(grid, nz=1, period=1.0, maxshift=0.4):
    nx = grid["nx"]  # [0]
    ny = grid["ny"]  # [0]
    zshift = grid["zShift"]
    Rxy = grid["Rxy"]
    Zxy = grid["Zxy"]

    # dz = 2.0 * pi / (period * (nz - 1))
    phi0 = np.linspace(0, 2.0 * pi / period, nz)

    # Need to insert additional points in Y so mesh looks smooth
    # for y in range(1,ny):
    #    ms = np.max(np.abs(zshift[:,y] - zshift[:,y-1]))
    #    if(

    # Create array of points, structured

    points = np.zeros([nx * ny * nz, 3])

    start = 0
    for y in range(ny):

        end = start + nx * nz

        phi = zshift[:, y] + phi0[:, None]
        r = Rxy[:, y] + (np.zeros([nz]))[:, None]

        xz_points = points[start:end]

        xz_points[:, 0] = (r * cos(phi)).ravel()  # X
        xz_points[:, 1] = (r * sin(phi)).ravel()  # Y
        xz_points[:, 2] = (Zxy[:, y] + (np.zeros([nz]))[:, None]).ravel()  # Z

        start = end

    return points


def create_grid(grid, data, period=1):
    """
    Create a structured grid which can be plotted with Mayavi
    or saved to a VTK file.

    Example
    -------
    from boutdata.collect import collect
    from boututils.file_import import file_import
    from boututile import boutgrid

    # Load grid file
    g = file_import("bout.grd.nc")

    # Load 3D data (x,y,z)
    data = collect("P", tind=-1)[0,:,:,:]

    # Create a structured grid
    sgrid = boutgrid.create_grid(g, data, 1)

    # Write structured grid to file
    w = tvtk.XMLStructuredGridWriter(input=sgrid, file_name='sgrid.vts')
    w.write()

    # View the structured grid
    boutgrid.view3d(sgrid)
    """
    s = np.shape(data)

    nx = grid["nx"]  # [0]
    ny = grid["ny"]  # [0]
    nz = s[2]

    print("data: %d,%d,%d   grid: %d,%d\n" % (s[0], s[1], s[2], nx, ny))

    dims = (nx, nz, ny)
    sgrid = tvtk.StructuredGrid(dimensions=dims)
    pts = aligned_points(grid, nz, period)
    print(np.shape(pts))
    sgrid.points = pts

    scalar = np.zeros([nx * ny * nz])
    start = 0
    for y in range(ny):
        end = start + nx * nz

        # scalar[start:end] = (data[:,y,:]).transpose().ravel()
        scalar[start:end] = (data[:, y, :]).ravel()

        print(y, " = ", np.max(scalar[start:end]))
        start = end

    sgrid.point_data.scalars = np.ravel(scalar.copy())
    sgrid.point_data.scalars.name = "data"

    return sgrid


# @mayavi2.standalone
def view3d(sgrid):
    from mayavi.api import Engine
    from mayavi.core.ui.engine_view import EngineView
    from mayavi.modules.api import GridPlane, Outline
    from mayavi.sources.vtk_data_source import VTKDataSource

    e = Engine()
    e.start()
    _ = e.new_scene()
    # Do this if you need to see the MayaVi tree view UI.
    ev = EngineView(engine=e)
    _ = ev.edit_traits()

    #    mayavi.new_scene()
    src = VTKDataSource(data=sgrid)
    e.add_source(src)
    e.add_module(Outline())
    g = GridPlane()
    g.grid_plane.axis = "x"
    e.add_module(g)
