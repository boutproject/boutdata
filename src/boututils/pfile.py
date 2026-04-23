"""A class object used to interface with Osborne pfiles."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ["pfile"]
__date__ = "11152017"
__version__ = "1.0.1"

import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si

# from boututils.fileio import save2nc


class pfile(OrderedDict):
    """A class object used to interface with Osborne pfiles.

    This class is inheritated from `collections.OrderedDict <https://docs.
    python.org/2/library/collections.html#collections.OrderedDict>`_. Each
    variable are loaded as an dict item, its value is an dict object including
    keys 'data', 'description', 'units', 'psinorm', 'derivative'.

    Attributes
    ----------
    legacy : bool, default: False
        If ``legacy=False``, it means the pfile is a new-version which
        includes the information of impurities.
    filename : str
        File name of the pfile.
    discriptions : dict
        An dict contains discriptions string about each variable in the pfile.
    units : dict
        An dict contains units for each variable in the pfile.

    Methods
    -------
    load()
        load pfile data, called at the initiation stage.
    save(filename=None)
        save pfile data to ``filename`` if it is given. Otherwise overwrite
        the original file.
    plot()
        plot profiles in the pfile.
    interpolate()
        interpolate the profiles to the same psinorm. Return all profiles and
        ``psin``.

    Notes
    -----
    This class is modified from `omfit_osborne.py`_ in the `OMFIT project`_.

    .. _omfit_osborne.py: http://gafusion.github.io/OMFIT-source/_modules/
        classes/omfit_osborne.html#OMFITosborneProfile
    .. _OMFIT project: http://gafusion.github.io/OMFIT-source/index.html

    """

    def __init__(self, filename):
        OrderedDict.__init__(self)

        self.legacy = False
        self.filename = filename.split("/")[-1]
        self.filename_full = filename

        self.descriptions = OrderedDict()
        self.descriptions["ne"] = "Electron density"
        self.descriptions["te"] = "Electron temperature"
        self.descriptions["ni"] = "Ion density"
        self.descriptions["ti"] = "Ion temperature"
        self.descriptions["nb"] = "Fast ion density"
        self.descriptions["pb"] = "Fast ion pressure"
        self.descriptions["ptot"] = "Total pressure"
        self.descriptions["omeg"] = "Toroidal rotation: VTOR/R"
        self.descriptions["omegp"] = "Poloidal rotation: Bt * VPOL / (RBp)"
        self.descriptions["omgvb"] = (
            "VxB rotation term in the ExB rotation frequency: OMEG + OMEGP"
        )
        self.descriptions["omgpp"] = (
            "Diamagnetic term in the ExB rotation frequency: "
            "(P_Carbon)/dpsi / (6*n_Carbon)"
        )
        self.descriptions["omgeb"] = "ExB rotation frequency: OMGPP + OMGVB = Er/(RBp)"
        self.descriptions["er"] = (
            "Radial electric field from force balance: OMGEB * RBp"
        )
        self.descriptions["ommvb"] = (
            "Main ion VXB term of Er/RBp, considered a flux function"
        )
        self.descriptions["ommpp"] = (
            "Main ion pressure term of Er/RBp, considered a flux function"
        )
        self.descriptions["omevb"] = (
            "Electron VXB term of Er/RBp, considered a flux function"
        )
        self.descriptions["omepp"] = (
            "Electron pressure term of Er/RBp, considered a flux function"
        )
        self.descriptions["kpol"] = (
            "KPOL=VPOL/Bp : V_vector = KPOL*B_vector + OMGEB * PHI_Vector"
        )
        self.descriptions["N Z A"] = "N Z A of ION SPECIES"
        self.descriptions["omghb"] = (
            "Han-Burrell form for the ExB velocity shearing rate: "
            "OMGHB = (RBp)**2/Bt * d (Er/RBp)/dpsi"
        )
        self.descriptions["nz1"] = "Density of the 1st impurity species"
        self.descriptions["vtor1"] = "Toroidal velocity of the 1st impurity species"
        self.descriptions["vpol1"] = "Poloidal velocity of the 1st impurity species"
        self.descriptions["nz2"] = "Density of the 2nd impurity species"
        self.descriptions["vtor2"] = "Toroidal velocity of the 2nd impurity species"
        self.descriptions["vpol2"] = "Poloidal velocity of the 2nd impurity species"
        # There may be more impurity species, but let's stop here for now.

        self.units = OrderedDict()
        # The units will be updated according to the pfile
        self.units["ne"] = "10^20/m^3"
        self.units["te"] = "KeV"
        self.units["ni"] = "10^20/m^3"
        self.units["ti"] = "KeV"
        self.units["nb"] = "10^20/m^3"
        self.units["pb"] = "KPa"
        self.units["ptot"] = "KPa"
        self.units["omeg"] = "kRad/s"
        self.units["omegp"] = "kRad/s"
        self.units["omgvb"] = "kRad/s"
        self.units["omgpp"] = "kRad/s"
        self.units["omgeb"] = "kRad/s"
        self.units["ommvb"] = ""
        self.units["ommpp"] = ""
        self.units["omevb"] = ""
        self.units["omepp"] = ""
        self.units["er"] = "kV/m"
        self.units["kpol"] = "km/s/T"
        self.units["N Z A"] = ""
        self.units["omghb"] = ""
        self.units["nz1"] = "10^20/m^3"
        self.units["vtor1"] = "km/s"
        self.units["vpol1"] = "km/s"
        self.units["nz2"] = "10^20/m^3"
        self.units["vtor2"] = "km/s"
        self.units["vpol2"] = "km/s"

        self.load()

    def load(self):
        """Method used to load the content of the pfile."""

        # read the file
        if not os.path.getsize(self.filename_full):
            print("empty file!")
            return

        with open(self.filename_full, "r") as f:
            fl = f.read().strip().splitlines(True)

        ind = 0
        while True:
            try:
                line = fl[ind]
                line.split()
                fl[ind + 1].split()
            except IndexError:
                break

            pattern = "([0-9]*)\s(.*)\s(.*)\((.*)\)\s(.*)\n"
            num = int(re.sub("([0-9]*)\s.*", r"\1", line))
            xkey = re.sub(pattern, r"\2", line)
            key = re.sub(pattern, r"\3", line)
            units = re.sub(pattern, r"\4", line)
            der = re.sub(pattern, r"\5", line)

            quants = [xkey, key + "(" + units + ")", der]
            q = []
            for i in range(ind + 1, ind + 1 + num):
                q.append(map(float, fl[i].split()))
            q = list(zip(*q))
            if key in self.keys():
                if sum(np.array(self[key]["data"]) != np.array(q[1])):
                    raise (
                        ValueError(
                            "%s already defined, but trying to change "
                            "its value" % quants[1]
                        )
                    )
            elif "N Z A of ION SPECIES" in line:
                self["N Z A"] = {}
                self["N Z A"]["description"] = "N Z A of ION SPECIES"
                self["N Z A"]["N"] = np.array(q[0])
                self["N Z A"]["Z"] = np.array(q[1])
                self["N Z A"]["A"] = np.array(q[2])
            else:
                self[key] = {}
                if key in self.descriptions:
                    self[key]["description"] = self.descriptions[key]
                else:
                    self[key]["description"] = key
                self.units[key] = units
                self[key]["units"] = units
                for qi, quant in enumerate(quants):
                    if qi == 1:
                        self[key]["data"] = np.array(q[qi])
                    elif (quant == "d%s/dpsiN" % key) or (quant == "d%s/dpsi" % key):
                        self[key]["derivative"] = np.array(q[qi])
                    else:
                        self[key][quant] = np.array(q[qi])

            ind = ind + 1 + num

    def save(self, filename=None):
        """Save content of the object to the file in Osborne pfile format.

        Parameters
        ----------
        filename : str, optional
            if ``filename=None``, it will overwrite the original pfile.
            otherwise it will use ``filename`` as file name.

        """
        lines = []
        if filename is None:
            filename = self.filename_full

        for key in self.keys():
            if key == "N Z A":
                if self.legacy:
                    break
                lines.append("%i N Z A of ION SPECIES\n" % (len(self[key]["A"]),))
                for i in range(len(self[key]["A"])):
                    lines.append(
                        " %f %f %f\n"
                        % (self[key]["N"][i], self[key]["Z"][i], self[key]["A"][i])
                    )
            else:
                if len(self[key]["data"]) == 1:
                    continue
                lines.append(
                    "%i psinorm %s(%s) d%s/dpsiN\n"
                    % (len(self[key]["data"]), key, self[key]["units"], key)
                )
                for i in range(len(self[key]["data"])):
                    lines.append(
                        " %f   %f   %f\n"
                        % (
                            self[key]["psinorm"][i],
                            self[key]["data"][i],
                            self[key]["derivative"][i],
                        )
                    )

        with open(filename, "w") as f:
            f.writelines(lines)

    def interpolate(self, npoints=500, psin=None, output=False):
        """Interpolate profiles to same psin array.

        Parameters
        ----------
        npoints : int, default: 500
            number of data points in interval [0, 1.0], if ``psin==None``.
        psin : ndarray, optional, default: None
            interpolate profiles with given psin.
        output : bool or str, optional, default: False
            save data to netcdf file.
            if True, save data to file "`pfilename`_nx`npoints`.nc".
            if it is an string, then save data to file "`output`.nc"

        Returns
        -------
        data : dict
            profiles with same psin.

        """
        data = OrderedDict()
        if psin is None:
            psin = np.linspace(0, 1, npoints)
        else:
            if np.max(psin) > 1.0 or np.min(psin) < 0.0:
                raise ValueError("psin outside of [0, 1.0] is not supported!")
            npoints = np.size(psin)

        data["psin"] = psin
        for key in self.keys():
            if key == "N Z A":
                print("WARNING: information about ION SPECIES ignored!")
                continue
            data[key] = si.interp1d(
                self[key]["psinorm"], self[key]["data"], kind="cubic"
            )(psin)

        if isinstance(output, bool):
            outputnc = "{}_nx{}".format(self.filename, npoints)
        elif isinstance(output, str):
            outputnc = output

        if output:
            print("saving data to \n\t{} ...".format(outputnc))
            # save2nc(outputnc, 'a', **data)
            print("WARNING: This line was commented out!")

        return data

    def plot(self):
        """Method used to plot all profiles, one in a different subplots."""
        nplot = len(self)
        cplot = np.floor(np.sqrt(nplot) / 2.0) * 2
        rplot = np.ceil(nplot * 1.0 / cplot)
        plt.subplots_adjust(wspace=0.35, hspace=0.0)
        plt.ioff()
        try:
            for k, item in enumerate(self):
                r = np.floor(k * 1.0 / cplot)
                c = k - r * cplot

                if k == 0:
                    ax1 = plt.subplot(rplot, cplot, r * (cplot) + c + 1)
                    ax = ax1
                else:
                    ax = plt.subplot(rplot, cplot, r * (cplot) + c + 1, sharex=ax1)
                ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                if "psinorm" not in self[item]:
                    print("Can't plot {} because no psinorm attribute".format(item))
                    continue
                x = self[item]["psinorm"]

                plt.plot(x, self[item]["data"], "-")
                plt.text(
                    0.5,
                    0.9,
                    item,
                    horizontalalignment="center",
                    verticalalignment="top",
                    size="medium",
                    transform=ax.transAxes,
                )

                if k < len(self) - cplot:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    plt.xlabel("$\psi$")

                plt.xlim(min(x), max(x))

        finally:
            plt.draw()
            plt.ion()
