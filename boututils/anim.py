"""Animate graph with mayavi"""


import os
from builtins import range

import numpy as np
from boutdata.collect import collect

try:
    from enthought.mayavi import mlab
    from enthought.mayavi.mlab import colorbar, contour_surf, surf
except ImportError:
    try:
        from mayavi import mlab
        from mayavi.mlab import colorbar, contour_surf, surf
    except ImportError:
        print("No mlab available")

from tvtk.tools import visual


@mlab.show
@mlab.animate(delay=250)
def anim(s, d, *args, **kwargs):
    """Animate graph with mayavi

    Parameters
    ----------
    s : mayavi axis object
        Axis to animate data on
    d : array_like
        3-D array to animate
    s1 : mayavi axis object, optional
        Additional bundled graph (first item in *args)
    save : bool, optional
        Save png files for creating movie (default: False)

    """

    if len(args) == 1:
        s1 = args[0]
    else:
        s1 = None

    save = kwargs.get("save", False)

    nt = d.shape[0]

    print("animating for ", nt, "timesteps")
    if save:
        print("Saving pics in folder Movie")
        if not os.path.exists("Movie"):
            os.makedirs("Movie")

    for i in range(nt):
        s.mlab_source.scalars = d[i, :, :]
        if s1 is not None:
            s1.mlab_source.scalars = d[i, :, :]
        title = "t=" + np.string0(i)
        mlab.title(title, height=1.1, size=0.26)
        if save:
            mlab.savefig("Movie/anim%d.png" % i)
        yield


if __name__ == "__main__":

    path = "../../../examples/elm-pb/data"

    data = collect("P", path=path)

    nt = data.shape[0]

    ns = data.shape[1]
    ne = data.shape[2]
    nz = data.shape[3]

    f = mlab.figure(size=(600, 600))
    # Tell visual to use this as the viewer.
    visual.set_viewer(f)

    # First way

    s1 = contour_surf(
        data[0, :, :, 10] + 0.1, contours=30, line_width=0.5, transparent=True
    )
    s = surf(
        data[0, :, :, 10] + 0.1, colormap="Spectral"
    )  # , warp_scale='.1')#, representation='wireframe')

    # second way

    # x, y= mgrid[0:ns:1, 0:ne:1]
    # s = mesh(x,y,data[0,:,:,10], colormap='Spectral')#, warp_scale='auto')
    # #, representation='wireframe')
    s.enable_contours = True
    s.contour.filled_contours = True
    #

    # x, y, z= mgrid[0:ns:1, 0:ne:1, 0:nz:1]
    #
    # p=plot3d(x,y,z,data[10,:,:,:], tube_radius=0.025, colormap='Spectral')
    # p=points3d(x,y,z,data[10,:,:,:], colormap='Spectral')
    #
    # s=contour3d(x,y,z,data[10,:,:,:], contours=4, transparent=True)

    # mlab.view(0.,0.)
    colorbar()
    # axes()
    # outline()

    # Run the animation.
    anim(s, data[:, :, :, 10] + 0.1, s1, save=True)
