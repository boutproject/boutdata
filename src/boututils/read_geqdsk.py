import numpy

from .bunch import Bunch
from .geqdsk import Geqdsk


def read_geqdsk(filename: str):
    data = Geqdsk(filename)

    nxefit = data["nw"]
    nyefit = data["nh"]
    xdim = data["rdim"]
    zdim = data["zdim"]
    rgrid1 = data["rleft"]
    zmid = data["zmid"]

    nlim = data["limitr"]

    if nlim != 0:
        xlim = data["rlim"]
        ylim = data["zlim"]
    else:
        xlim = [0]
        ylim = [0]

    # Reconstruct the (R,Z) mesh
    r = numpy.zeros((nxefit, nyefit), numpy.float64)
    z = numpy.zeros((nxefit, nyefit), numpy.float64)

    for i in range(0, nxefit):
        for j in range(0, nyefit):
            r[i, j] = rgrid1 + xdim * i / (nxefit - 1)
            z[i, j] = (zmid - 0.5 * zdim) + zdim * j / (nyefit - 1)

    return Bunch(
        nx=nxefit,
        ny=nyefit,  # Number of horizontal and vertical points
        r=r,
        z=z,  # Location of the grid-points
        xdim=xdim,
        zdim=zdim,  # Size of the domain in meters
        rcentr=data["rcentr"],
        bcentr=data["bcentr"],  # Reference vacuum toroidal field (m, T)
        rgrid1=rgrid1,  # R of left side of domain
        zmid=zmid,  # Z at the middle of the domain
        rmagx=data["rmagx"],
        zmagx=data["zmagx"],  # Location of magnetic axis
        simagx=data["simagx"],  # Poloidal flux at the axis (Weber / rad)
        sibdry=data["sibdry"],  # Poloidal flux at plasma boundary (Weber / rad)
        cpasma=data["current"],  #
        psi=data["psirz"].T,  # Poloidal flux in Weber/rad on grid points
        fpol=data["fpol"],  # Poloidal current function on uniform flux grid
        pres=data["pres"],  # Plasma pressure in nt/m^2 on uniform flux grid
        qpsi=data["qpsi"],  # q values on uniform flux grid
        nbdry=data["nbbbs"],
        rbdry=data["rbbbs"],
        zbdry=data["zbbbs"],  # Plasma boundary
        nlim=nlim,
        xlim=xlim,
        ylim=ylim,
    )
