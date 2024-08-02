"""
G-Eqdsk reader class

The official document describing g-eqdsk files:
https://fusion.gat.com/conferences/snowmass/working/mfe/physics/p3/equilibria/g_eqdsk_s.pdf
"""

import warnings

import freeqdsk
import numpy

from .bunch import Bunch


class Geqdsk:
    def __init__(self, filename: str):
        """
        Constructor
        """
        warnings.warn(
            "`boututils.geqdsk.Geqdsk` is deprecated, use `freeqdsk.geqdsk.read` directly",
            DeprecationWarning,
        )

        self.data = {}

        self.openFile(filename)

    def openFile(self, filename):
        """
        open geqdsk file and parse its content
        """

        with open(filename, "r") as f:
            data = freeqdsk.geqdsk.read(f)

        self.data = {
            "case": (data["comment"], "Identification character string"),
            "nw": (data["nw"], "Number of horizontal R grid points"),
            "nh": (data["nh"], "Number of vertical Z grid points"),
            "rdim": (
                data["rdim"],
                "Horizontal dimension in meter of computational box",
            ),
            "zdim": (
                data["zdim"],
                "Vertical dimension in meter of computational box",
            ),
            "rcentr": (
                data["rcentr"],
                "R in meter of vacuum toroidal magnetic field BCENTR",
            ),
            "rleft": (
                data["rleft"],
                "Minimum R in meter of rectangular computational box",
            ),
            "zmid": (
                data["zmid"],
                "Z of center of computational box in meter",
            ),
            "rmaxis": (data["rmagx"], "R of magnetic axis in meter"),
            "zmaxis": (data["zmagx"], "Z of magnetic axis in meter"),
            "simag": (
                data["simagx"],
                "poloidal flux at magnetic axis in Weber /rad",
            ),
            "sibry": (
                data["sibdry"],
                "poloidal flux at the plasma boundary in Weber /rad",
            ),
            "bcentr": (
                data["bcentr"],
                "Vacuum toroidal magnetic field in Tesla at RCENTR",
            ),
            "current": (data["cpasma"], "Plasma current in Ampere"),
            "fpol": (
                data["fpol"],
                "Poloidal current function in m-T, F = RBT on flux grid",
            ),
            "pres": (
                data["pres"],
                "Plasma pressure in nt / m 2 on uniform flux grid",
            ),
            "ffprime": (
                data["ffprime"],
                "FF'(psi), in (mT),^2/(Weber/rad), on uniform flux grid",
            ),
            "pprime": (
                data["pprime"],
                "P'(psi), in (nt/m2),/(Weber/rad), on uniform flux grid",
            ),
            "psirz": (
                data["psi"],
                "Poloidal flux in Weber / rad on the rectangular grid points",
            ),
            "qpsi": (
                data["qpsi"],
                "q values on uniform flux grid from axis to boundary",
            ),
            "nbbbs": (data["nbdry"], "Number of boundary points"),
            "limitr": (data["nlim"], "Number of limiter points"),
            "rbbbs": (
                data["rbdry"],
                "R of boundary points in meter",
            ),
            "zbbbs": (
                data["zbdry"],
                "Z of boundary points in meter",
            ),
            "rlim": (
                data["rlim"],
                "R of surrounding limiter contour in meter",
            ),
            "zlim": (
                data["zlim"],
                "Z of surrounding limiter contour in meter",
            ),
        }

    def getAll(self):
        return self.data

    def getAllVars(self):
        return list(self.data.keys())

    def get(self, varname):
        return self.data[varname.lower()][0]

    def __getitem__(self, item):
        return self.get(item)

    def getDescriptor(self, varname):
        return self.data[varname.lower()][1]


def read_geqdsk(filename: str):
    warnings.warn(
        "`boututils.geqdks.read_geqdsk` is deprecated, use `freeqdsk.geqdsk.read` directly",
        DeprecationWarning,
    )

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


################################


def main():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="g-eqdsk file", default="")
    parser.add_option(
        "-a",
        "--all",
        dest="all",
        help="display all variables",
        action="store_true",
    )
    parser.add_option(
        "-v",
        "--vars",
        dest="vars",
        help="comma separated list of variables (use '-v \"*\"' for all)",
        default="*",
    )
    parser.add_option(
        "-p",
        "--plot",
        dest="plot",
        help="plot all variables",
        action="store_true",
    )
    parser.add_option(
        "-i",
        "--inquire",
        dest="inquire",
        help="inquire list of variables",
        action="store_true",
    )

    options, args = parser.parse_args()
    if not options.filename:
        parser.error("MUST provide filename (type -h for list of options)")

    geq = Geqdsk()
    geq.openFile(options.filename)

    if options.inquire:
        print(geq.getAllVars())

    if options.all:
        print(geq.getAll())

    vs = geq.getAllVars()
    if options.vars != "*":
        vs = options.vars.split(",")

    for v in vs:
        print(f"{v}: {geq.get(v)}")

    if options.plot:
        from matplotlib import pylab

        if options.vars == "*":
            options.vars = geq.getAllVars()
            print(options.vars)
        else:
            vs = options.vars.split(",")
            options.vars = vs

        xmin = geq.get("simag")
        xmax = geq.get("sibry")
        nx = geq.get("nw")
        dx = float(xmax - xmin) / float(nx - 1)
        x = numpy.arange(xmin, xmin + (xmax - xmin) * (1.0 + 1.0e-6), dx)
        for v in options.vars:
            if v[0] != "r" and v[0] != "z":
                data = geq.get(v)
                if len(numpy.shape(data)) == 1:
                    pylab.figure()
                    pylab.plot(x, data)
                    pylab.xlabel("psi poloidal")
                    pylab.ylabel(v)
                    pylab.title(geq.getDescriptor(v))
        # 2d plasma plot
        nw = geq.get("nw")
        nh = geq.get("nh")
        rmin = geq.get("rleft")
        rmax = rmin + geq.get("rdim")
        dr = float(rmax - rmin) / float(nw - 1)
        zmin = geq.get("zmid") - geq.get("zdim") / 2.0
        zmax = geq.get("zmid") + geq.get("zdim") / 2.0
        dz = (zmax - zmin) / float(nh - 1)
        rs = numpy.arange(rmin, rmin + (rmax - rmin) * (1.0 + 1.0e-10), dr)
        zs = numpy.arange(zmin, zmin + (zmax - zmin) * (1.0 + 1.0e-10), dz)
        pylab.figure()
        pylab.pcolor(rs, zs, geq.get("psirz"), shading="interp")
        pylab.plot(geq.get("rbbbs"), geq.get("zbbbs"), "w-")
        pylab.axis("image")
        pylab.title("poloidal flux")
        pylab.xlabel("R")
        pylab.ylabel("Z")

        pylab.show()


if __name__ == "__main__":
    main()
