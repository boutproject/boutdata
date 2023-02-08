"""Routines for manipulating restart files

TODO
----

- Don't import ``numpy.random.normal`` directly, just the ``random``
  submodule, or sphinx includes the documentation for ``normal``

"""

import os
import glob

from boutdata.collect import collect, create_cache
from boututils.datafile import DataFile
from boututils.boutarray import BoutArray
from boutdata.processor_rearrange import get_processor_layout, create_processor_layout

import multiprocessing
from natsort import natsorted
import numpy as np
from numpy import mean, zeros, arange
from numpy.random import normal

from scipy.interpolate import interp1d

from boutdata import shiftz

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    pass


def resize3DField(var, data, coordsAndSizesTuple, method, mute):
    """Resize 3D fields

    To be called by resize.

    Written as a function in order to call it using multiprocess. Must
    be defined as a top level function in order to be pickable by the
    multiprocess.

    See the function resize for details

    """

    # Unpack the tuple for better readability
    (
        xCoordOld,
        yCoordOld,
        zCoordOld,
        xCoordNew,
        yCoordNew,
        zCoordNew,
        newNx,
        newNy,
        newNz,
    ) = coordsAndSizesTuple

    if not (mute):
        print(
            "    Resizing "
            + var
            + " to (nx,ny,nz) = ({},{},{})".format(newNx, newNy, newNz)
        )

    # Make the regular grid function (see examples in
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    # for details)
    gridInterpolator = RegularGridInterpolator(
        (xCoordOld, yCoordOld, zCoordOld), data, method
    )

    # Need to fill with one exrta z plane (will only contain zeros)
    newData = np.zeros((newNx, newNy, newNz))

    # Interpolate to the new values
    for xInd, x in enumerate(xCoordNew):
        for yInd, y in enumerate(yCoordNew):
            for zInd, z in enumerate(zCoordNew):
                newData[xInd, yInd, zInd] = gridInterpolator([x, y, z])

    return var, newData


def resize(
    newNx,
    newNy,
    newNz,
    mxg=2,
    myg=2,
    path="data",
    output="./",
    informat="nc",
    outformat=None,
    method="linear",
    maxProc=None,
    mute=False,
):
    """Increase/decrease the number of points in restart files.

    NOTE: Can't overwrite
    WARNING: Currently only implemented with uniform BOUT++ grid

    Parameters
    ----------
    newNx, newNy, newNz : int
        nx, ny, nz for the new file (including ghost points)
    mxg, myg : int, optional
        Number of ghost points in x, y (default: 2)
    path : str, optional
        Input path to data files
    output : str, optional
        Path to write new files
    informat : str, optional
        File extension of input
    outformat : {None, str}, optional
        File extension of output (default: use the same as `informat`)
    method : {'linear', 'nearest'}, optional
        What interpolation method to be used
    maxProc : {None, int}, optional
        Limits maximum processors to use when interpolating if set
    mute : bool, optional
        Whether or not output should be printed from this function

    Returns
    -------
    return : bool
        True on success, else False

    TODO
    ----
    - Add 2D field interpolation
    - Replace printing errors with raising `ValueError`
    - Make informat work like `redistribute`

    """

    if method is None:
        # Make sure the method is set
        method = "linear"

    if outformat is None:
        outformat = informat

    if path == output:
        print("ERROR: Can't overwrite restart files when expanding")
        return False

    def is_pow2(x):
        """Returns true if x is a power of 2"""
        return (x > 0) and ((x & (x - 1)) == 0)

    if not is_pow2(newNz):
        print("ERROR: New Z size {} must be a power of 2".format(newNz))
        return False

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*." + informat))
    file_list.sort()
    nfiles = len(file_list)

    if nfiles == 0:
        print("ERROR: No data found in {}".format(path))
        return False

    if not (mute):
        print("Number of files found: " + str(nfiles))

    for f in file_list:
        new_f = os.path.join(output, f.split("/")[-1])
        if not (mute):
            print("Changing {} => {}".format(f, new_f))

        # Open the restart file in read mode and create the new file
        with DataFile(f) as old, DataFile(new_f, write=True, create=True) as new:

            # Find the dimension
            for var in old.list():
                # Read the data
                data = old.read(var)
                # Find 3D variables
                if old.ndims(var) == 3:
                    break

            nx, ny, nz = data.shape
            # Make coordinates
            # NOTE: The max min of the coordinates are irrelevant when
            #       interpolating (as long as old and new coordinates
            #       are consistent), so we just choose all variable to
            #       be between 0 and 1 Calculate the old coordinates
            xCoordOld = np.linspace(0, 1, nx)
            yCoordOld = np.linspace(0, 1, ny)
            zCoordOld = np.linspace(0, 1, nz)

            # Calculate the new coordinates
            xCoordNew = np.linspace(xCoordOld[0], xCoordOld[-1], newNx)
            yCoordNew = np.linspace(yCoordOld[0], yCoordOld[-1], newNy)
            zCoordNew = np.linspace(zCoordOld[0], zCoordOld[-1], newNz)

            # Make a pool of workers
            pool = multiprocessing.Pool(maxProc)
            # List of jobs and results
            jobs = []
            # Pack input to resize3DField together
            coordsAndSizesTuple = (
                xCoordOld,
                yCoordOld,
                zCoordOld,
                xCoordNew,
                yCoordNew,
                zCoordNew,
                newNx,
                newNy,
                newNz,
            )

            # Loop over the variables in the old file
            for var in old.list():
                # Read the data
                data = old.read(var)
                attributes = old.attributes(var)

                # Find 3D variables
                if old.ndims(var) == 3:

                    # Asynchronous call (locks first at .get())
                    jobs.append(
                        pool.apply_async(
                            resize3DField,
                            args=(
                                var,
                                data,
                                coordsAndSizesTuple,
                                method,
                                mute,
                            ),
                        )
                    )

                else:
                    if not (mute):
                        print("    Copying " + var)
                        newData = data.copy()
                    if not (mute):
                        print("Writing " + var)
                    new.write(var, newData)

            for job in jobs:
                var, newData = job.get()
                newData = BoutArray(newData, attributes=attributes)
                if not (mute):
                    print("Writing " + var)
                new.write(var, newData)

            # Close the pool of workers
            pool.close()
            # Wait for all processes to finish
            pool.join()

    return True


def resizeZ(newNz, path="data", output="./", informat="nc", outformat=None):
    """Increase the number of Z points in restart files

    NOTE:
        * Can't overwrite
        * Will not yield a result close to the original if there are
          asymmetries in the z-direction

    Parameters
    ----------
    newNz : int
        nz for the new file
    path : str, optional
        Path to original restart files (default: "data")
    output : str, optional
        Path to write new restart files (default: current directory)
    informat : str, optional
        File extension of original files (default: "nc")
    outformat : str, optional
        File extension of new files (default: use the same as `informat`)

    Returns
    -------
    True on success, else False

    TODO
    ----
    - Replace printing errors with raising `ValueError`
    - Make informat work like `redistribute`

    """

    if outformat is None:
        outformat = informat

    if path == output:
        raise ValueError("Can't overwrite restart files when expanding")

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*." + informat))
    file_list.sort()
    nfiles = len(file_list)

    if nfiles == 0:
        raise ValueError("No data found")

    print("Number of files found: " + str(nfiles))

    for f in file_list:
        new_f = os.path.join(output, f.split("/")[-1])
        print("Changing {} => {}".format(f, new_f))

        # Open the restart file in read mode and create the new file
        with DataFile(f) as old, DataFile(new_f, write=True, create=True) as new:
            # Loop over the variables in the old file
            for var in old.list():
                # Read the data
                data = old.read(var)
                attributes = old.attributes(var)

                # Find 3D variables
                if old.ndims(var) == 3:
                    print("    Resizing " + var)

                    nx, ny, nz = data.shape

                    newdata = np.zeros((nx, ny, newNz))
                    for x in range(nx):
                        for y in range(ny):
                            f_old = np.fft.fft(data[x, y, :])

                            # Number of points in f is power of 2
                            f_new = np.zeros(newNz)

                            # Copy coefficients across (ignoring Nyquist)
                            f_new[0] = f_old[0]  # DC
                            for m in range(1, int(nz / 2)):
                                # + ve frequencies
                                f_new[m] = f_old[m]
                                # - ve frequencies
                                f_new[newNz - m] = f_old[nz - m]

                            # Invert fft
                            newdata[x, y, :] = np.fft.ifft(f_new).real
                            newdata[x, y, :] = newdata[x, y, 0]

                    # Multiply with the ratio of newNz/nz
                    # This is not needed in the IDL routine as the
                    # forward transfrom has the scaling factor 1/N in
                    # the forward transform, whereas the scaling factor
                    # 1/N is the inverse transform in np.fft
                    # Note that ifft(fft(a)) = a for the same number of
                    # points in both IDL and np.ftt
                    newdata *= newNz / nz
                elif var == "nz":
                    print("    Changing " + var)
                    newdata = newNz
                else:
                    print("    Copying " + var)
                    newdata = data.copy()

                newdata = BoutArray(newdata, attributes=attributes)

                new.write(var, newdata)

    return True


def addnoise(path=".", var=None, scale=1e-5):
    """Add random noise to restart files

    .. warning:: Modifies restart files in place! This is in contrast
                 to most of the functions in this module!

    Parameters
    ----------
    path : str, optional
        Path to restart files (default: current directory)
    var : str, optional
        The variable to modify. By default all 3D variables are modified
    scale : float
        Amplitude of the noise. Gaussian noise is used, with zero mean
        and this parameter as the standard deviation

    """
    file_list = glob.glob(os.path.join(path, "BOUT.restart.*"))
    nfiles = len(file_list)

    print("Number of restart files: %d" % (nfiles,))

    for file in file_list:
        print(file)
        with DataFile(file, write=True) as d:
            if var is None:
                for v in d.list():
                    if d.ndims(v) == 3:
                        print(" -> " + v)
                        data = d.read(v, asBoutArray=True)
                        data += normal(scale=scale, size=data.shape)
                        d.write(v, data)
            else:
                # Modify a single variable
                print(" -> " + var)
                data = d.read(var)
                data += normal(scale=scale, size=data.shape)
                d.write(var, data)


def scalevar(var, factor, path="."):
    """Scales a variable by a given factor, modifying restart files in
    place

    .. warning:: Modifies restart files in place! This is in contrast
                 to most of the functions in this module!

    Parameters
    ----------
    var : str
        Name of the variable
    factor : float
        Factor to multiply
    path : str, optional
        Path to the restart files (default: current directory)

    """

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*"))
    nfiles = len(file_list)

    print("Number of restart files: %d" % (nfiles,))
    for file in file_list:
        print(file)
        with DataFile(file, write=True) as d:
            d[var] = d[var] * factor


def create(
    averagelast=1, final=-1, path="data", output="./", informat="nc", outformat=None
):
    """Create restart files from data (dmp) files.

    Parameters
    ----------
    averagelast : int, optional
        Number of time points (counting from `final`, inclusive) to
        average over (default is 1 i.e. just take last time-point)
    final : int, optional
        The last time point to use (default is last, -1)
    path : str, optional
        Path to original restart files (default: "data")
    output : str, optional
        Path to write new restart files (default: current directory)
    informat : str, optional
        File extension of original files (default: "nc")
    outformat : str, optional
        File extension of new files (default: use the same as `informat`)

    """

    if outformat is None:
        outformat = informat

    file_list = glob.glob(os.path.join(path, "BOUT.dmp.*." + informat))
    nfiles = len(file_list)

    print(("Number of data files: ", nfiles))

    for i in range(nfiles):
        # Open each data file
        infname = os.path.join(path, "BOUT.dmp." + str(i) + "." + informat)
        outfname = os.path.join(output, "BOUT.restart." + str(i) + "." + outformat)

        print((infname, " -> ", outfname))

        infile = DataFile(infname)
        outfile = DataFile(outfname, create=True)

        # Get the data always needed in restart files
        hist_hi = infile.read("iteration")
        print(("hist_hi = ", hist_hi))
        outfile.write("hist_hi", hist_hi)

        t_array = infile.read("t_array")
        tt = t_array[final]
        print(("tt = ", tt))
        outfile.write("tt", tt)

        tind = final
        if tind < 0.0:
            tind = len(t_array) + final

        NXPE = infile.read("NXPE")
        NYPE = infile.read("NYPE")
        print(("NXPE = ", NXPE, " NYPE = ", NYPE))
        outfile.write("NXPE", NXPE)
        outfile.write("NYPE", NYPE)

        # Get a list of variables
        varnames = infile.list()

        for var in varnames:
            if infile.ndims(var) == 4:
                # Could be an evolving variable

                print((" -> ", var))

                data = infile.read(var)

                if averagelast == 1:
                    slice = data[final, :, :, :]
                else:
                    slice = mean(data[(final - averagelast) : final, :, :, :], axis=0)

                print(slice.shape)

                outfile.write(var, slice)

        infile.close()
        outfile.close()


def redistribute(
    npes,
    path="data",
    nxpe=None,
    output=".",
    informat=None,
    outformat=None,
    mxg=None,
    myg=None,
):
    """Resize restart files across NPES processors.

    Does not check if new processor arrangement is compatible with the
    branch cuts. In this respect :py:func:`restart.split` is
    safer. However, BOUT++ checks the topology during initialisation
    anyway so this is not too serious.

    Parameters
    ----------
    npes : int
        Number of processors for the new restart files
    path : str, optional
        Path to original restart files (default: "data")
    nxpe : int, optional
        Number of processors to use in the x-direction (determines
        split: npes = nxpe * nype). Default is None which uses the
        same algorithm as BoutMesh (but without topology information)
        to determine a suitable value for nxpe.
    output : str, optional
        Location to save new restart files (default: current directory)
    informat : str, optional
        Specify file format of old restart files (must be a suffix
        understood by DataFile, e.g. 'nc'). Default uses the format of
        the first 'BOUT.restart.*' file listed by glob.glob.
    outformat : str, optional
        Specify file format of new restart files (must be a suffix
        understood by DataFile, e.g. 'nc'). Default is to use the same
        as informat.
    mxg, myg : int, optional
        Number of guard cells in x, y to use in the output (default:
        keep the same as input)

    Returns
    -------
    True on success

    TODO
    ----
    - Replace printing errors with raising `ValueError`

    """

    if npes <= 0:
        print("ERROR: Negative or zero number of processors")
        return False

    if path == output:
        print("ERROR: Can't overwrite restart files")
        return False

    if informat is None:
        file_list = glob.glob(os.path.join(path, "BOUT.restart.*"))
    else:
        file_list = glob.glob(os.path.join(path, "BOUT.restart.*." + informat))

    nfiles = len(file_list)

    # Read old processor layout
    f = DataFile(file_list[0])

    # Get list of variables
    var_list = f.list()
    if len(var_list) == 0:
        print("ERROR: No data found")
        return False

    old_processor_layout = get_processor_layout(f, has_t_dimension=False)
    print(
        "Grid sizes: ",
        old_processor_layout.nx,
        old_processor_layout.ny,
        old_processor_layout.mz,
    )

    if nfiles != old_processor_layout.npes:
        print("WARNING: Number of restart files inconsistent with NPES")
        print("Setting nfiles = " + str(old_processor_layout.npes))
        nfiles = old_processor_layout.npes

    if nfiles == 0:
        print("ERROR: No restart files found")
        return False

    informat = file_list[0].split(".")[-1]
    if outformat is None:
        outformat = informat

    try:
        new_processor_layout = create_processor_layout(
            old_processor_layout, npes, nxpe=nxpe
        )
    except ValueError as e:
        print("Could not find valid processor split. " + e.what())

    old_mxg = old_processor_layout.mxg
    old_myg = old_processor_layout.myg

    if mxg is None:
        mxg = old_mxg
    if myg is None:
        myg = old_myg

    nxpe = new_processor_layout.nxpe
    nype = new_processor_layout.nype
    mxsub = new_processor_layout.mxsub
    mysub = new_processor_layout.mysub
    mzsub = new_processor_layout.mz

    if "jyseps2_1" not in f.keys():
        # Workaround for older output files that are missing jyseps* values
        is_doublenull = False
    else:
        jyseps2_1 = f["jyseps2_1"]
        ny_inner = f["ny_inner"]
        jyseps1_2 = f["jyseps1_2"]
        is_doublenull = jyseps2_1 == jyseps1_2

    outfile_list = []
    for i in range(npes):
        outpath = os.path.join(output, "BOUT.restart." + str(i) + "." + outformat)
        outfile_list.append(DataFile(outpath, write=True, create=True))

    DataFileCache = create_cache(path, "BOUT.restart")

    for v in var_list:
        dimensions = f.dimensions(v)

        # collect data
        data = collect(
            v, xguards=True, yguards=True, info=False, datafile_cache=DataFileCache
        )

        # write data
        for i in range(npes):
            ix = i % nxpe
            iy = int(i / nxpe)

            def get_block(data):
                if mxg > old_mxg or myg > old_myg:
                    # need to make a new array as some boundary cell points are not
                    # present in data
                    new_shape = list(data.shape)
                    new_shape[0] = mxsub + 2 * mxg
                    new_shape[1] = mysub + 2 * myg
                    result = np.zeros(new_shape)

                    if mxg > old_mxg:
                        d = mxg - old_mxg
                        result_slice_x = slice(d, -d)
                        data_slice_x = slice(ix * mxsub, (ix + 1) * mxsub + 2 * old_mxg)
                    else:
                        d = old_mxg - mxg
                        result_slice_x = slice(None)
                        data_slice_x = slice(
                            d + ix * mxsub, d + (ix + 1) * mxsub + 2 * mxg
                        )

                    if myg > old_myg:
                        d = myg - old_myg
                        result_slice_y = slice(d, -d)
                        data_slice_y = slice(iy * mysub, (iy + 1) * mysub + 2 * old_myg)
                    else:
                        d = old_myg - myg
                        result_slice_y = slice(None)
                        data_slice_y = slice(
                            d + iy * mysub, d + (iy + 1) * mysub + 2 * myg
                        )

                    result[result_slice_x, result_slice_y] = data[
                        data_slice_x, data_slice_y
                    ]

                    return result
                else:
                    xoffset = old_mxg - mxg
                    yoffset = old_myg - myg
                    return data[
                        xoffset + ix * mxsub : xoffset + (ix + 1) * mxsub + 2 * mxg,
                        yoffset + iy * mysub : yoffset + (iy + 1) * mysub + 2 * myg,
                    ]

            outfile = outfile_list[i]
            if v == "NPES":
                outfile.write(v, npes)
            elif v == "NXPE":
                outfile.write(v, nxpe)
            elif v == "NYPE":
                outfile.write(v, nype)
            elif v == "MXSUB":
                outfile.write(v, mxsub)
            elif v == "MYSUB":
                outfile.write(v, mysub)
            elif v == "MZSUB":
                outfile.write(v, mzsub)
            elif v == "MXG":
                outfile.write(v, mxg)
            elif v == "MYG":
                outfile.write(v, myg)
            elif v == "PE_XIND":
                outfile.write(v, ix)
            elif v == "PE_YIND":
                outfile.write(v, iy)
            elif dimensions == ("x", "y"):
                # Field2D
                outfile.write(v, get_block(data))
            elif dimensions == ("x", "z"):
                # FieldPerp
                yindex_global = data.attributes["yindex_global"]
                this_proc_yglobal_min = iy * mysub + myg
                this_proc_yglobal_max = (iy + 1) * mysub + myg
                if is_doublenull and this_proc_yglobal_max > ny_inner:
                    # Above upper x-point, so need to add offset for upper boundary
                    # cells
                    this_proc_yglobal_min += 2 * myg
                    this_proc_yglobal_max += 2 * myg
                if iy == 0 or (
                    is_doublenull and this_proc_yglobal_min == ny_inner + 3 * myg
                ):
                    # Has lower y-boundary, so include boundary cells in local yglobal
                    # range
                    this_proc_yglobal_min -= myg
                if iy == nype - 1 or (
                    is_doublenull and this_proc_yglobal_max == ny_inner + myg
                ):
                    # Has upper y-boundary, so include boundary cells in local yglobal
                    # range
                    this_proc_yglobal_max += myg
                if (
                    yindex_global >= this_proc_yglobal_min
                    and yindex_global < this_proc_yglobal_max
                ):
                    outfile.write(v, data[ix * mxsub : (ix + 1) * mxsub + 2 * mxg, :])
                else:
                    nullarray = BoutArray(
                        np.zeros([mxsub + 2 * mxg, mzsub]),
                        attributes={
                            "bout_type": "FieldPerp",
                            "yindex_global": -myg - 1,
                        },
                    )
                    outfile.write(v, nullarray)
            elif dimensions == ("x", "y", "z"):
                # Field3D
                outfile.write(v, get_block(data))
            elif not any(d in dimensions for d in ("x", "y", "z")):
                # scalar or other non-spatially-dependent variable
                outfile.write(v, data)
            else:
                print(
                    "ERROR: variable found with unexpected dimensions,", dimensions, v
                )

    f.close()
    for outfile in outfile_list:
        outfile.close()

    return True


def resizeY(newy, path="data", output=".", informat="nc", outformat=None, myg=2):
    """Increase the number of Y points in restart files

    NOTE:
        * Can't overwrite

    Parameters
    ----------
    newy : int
        ny for the new file
    path : str, optional
        Path to original restart files (default: "data")
    output : str, optional
        Path to write new restart files (default: current directory)
    informat : str, optional
        File extension of original files (default: "nc")
    outformat : str, optional
        File extension of new files (default: use the same as `informat`)
    myg : int, optional
        Number of ghost points in y (default: 2)

    Returns
    -------
    True on success, else False

    TODO
    ----
    - Replace printing errors with raising `ValueError`
    - Make informat work like `redistribute`

    """

    if outformat is None:
        outformat = informat

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*." + informat))

    nfiles = len(file_list)

    if nfiles == 0:
        print("ERROR: No restart files found")
        return False

    for i in range(nfiles):
        # Open each data file
        infname = os.path.join(path, "BOUT.restart." + str(i) + "." + informat)
        outfname = os.path.join(output, "BOUT.restart." + str(i) + "." + outformat)

        print("Processing %s -> %s" % (infname, outfname))

        infile = DataFile(infname)
        outfile = DataFile(outfname, create=True)

        # Copy basic information
        for var in ["hist_hi", "NXPE", "NYPE", "tt"]:
            data = infile.read(var)
            try:
                # Convert to scalar if necessary
                data = data[0]
            except (TypeError, IndexError):
                pass
            outfile.write(var, data)

        # Get a list of variables
        varnames = infile.list()

        for var in varnames:
            dimensions = infile.dimensions(var)
            if dimensions == ("x", "y", "z"):
                # Could be an evolving variable [x,y,z]

                print(" -> Resizing " + var)

                # Read variable from input
                indata = infile.read(var)

                nx, ny, nz = indata.shape

                # y coordinate in input and output data
                iny = (arange(ny) - myg + 0.5) / (ny - 2 * myg)
                outy = (arange(newy) - myg + 0.5) / (newy - 2 * myg)

                outdata = zeros([nx, newy, nz])

                for x in range(nx):
                    for z in range(nz):
                        f = interp1d(
                            iny, indata[x, :, z], bounds_error=False, fill_value=0.0
                        )
                        outdata[x, :, z] = f(outy)

                outfile.write(var, outdata)
            elif dimensions == ("x", "y"):
                # Assume evolving variable [x,y]
                print(" -> Resizing " + var)

                # Read variable from input
                indata = infile.read(var)

                nx, ny = indata.shape

                # y coordinate in input and output data
                iny = (arange(ny) - myg + 0.5) / (ny - 2 * myg)
                outy = (arange(newy) - myg + 0.5) / (newy - 2 * myg)

                outdata = zeros([nx, newy])

                for x in range(nx):
                    f = interp1d(iny, indata[x, :], bounds_error=False, fill_value=0.0)
                    outdata[x, :] = f(outy)

                outfile.write(var, outdata)
            else:
                # Copy variable
                print(" -> Copying " + var)

                # Read variable from input
                data = infile.read(var)
                try:
                    # Convert to scalar if necessary
                    data = data[0]
                except (TypeError, IndexError):
                    pass
                outfile.write(var, data)

        infile.close()
        outfile.close()


def addvar(var, value, path="."):
    """Adds a variable with constant value to all restart files.

    .. warning:: Modifies restart files in place! This is in contrast
                 to most of the functions in this module!

    This is useful for restarting simulations whilst turning on new
    equations. By default BOUT++ throws an error if an evolving
    variable is not in the restart file. By setting an option the
    variable can be set to zero. This allows it to start with a
    non-zero value.

    Parameters
    ----------
    var : str
        The name of the variable to add
    value : float
        Constant value for the variable
    path : str, optional
        Input path to data files (default: current directory)

    """

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*"))
    nfiles = len(file_list)

    print("Number of restart files: %d" % (nfiles,))
    # Loop through all the restart files
    for filename in file_list:
        print(filename)
        # Open the restart file for writing (modification)
        with DataFile(filename, write=True) as df:
            size = None
            # Find a 3D variable and get its size
            for varname in df.list():
                size = df.size(varname)
                if len(size) == 3:
                    break
            if size is None:
                raise Exception("no 3D variables found")

            # Create a new 3D array with input value
            data = np.zeros(size) + value

            # Set the variable in the NetCDF file
            df.write(var, data)


def change_grid(
    from_grid_file,
    to_grid_file,
    path="data",
    output=".",
    interpolator="nearest",
    show=False,
    floor=True,
):
    """
    Convert a set of restart files from one grid to another

    Notes:
    - Only working for 2D (axisymmetric) simulations with nz = 1
    - Does not support evolving Field2D or FieldPerp variables
    - Does not support grids with y boundary cells

    from_grid_file : str
         File containing the input grid
    to_grid_file : str
         File containing the output grid
    path : str, optional
         Directory containing input restart files
    output : str, optional
         Directory where output restart files will be written
    interpolator : str, optional
         Interpolation method to use. Options are 'nearest', 'CloughTocher', 'RBF', 'linear'
    show : bool, optional
         Display the interpolated fields using Matplotlib
    floor : bool, optional
         Apply floor to interpolated densities and pressures

    """

    # Read in grid files
    with DataFile(from_grid_file) as g:
        # Check for y boundary cells
        try:
            if g["y_boundary_guards"] != 0:
                raise ValueError(
                    "Support for grid files with y-boundary cells not implemented yet"
                )
        except KeyError:
            pass  # No y_boundary_guards key
        from_Rxy = g["Rxy"]
        from_Zxy = g["Zxy"]

    with DataFile(to_grid_file) as g:
        # Check for y boundary cells
        try:
            if g["y_boundary_guards"] != 0:
                raise ValueError(
                    "Support for grid files with y-boundary cells not implemented yet"
                )
        except KeyError:
            pass  # No y_boundary_guards key
        to_Rxy = g["Rxy"]
        to_Zxy = g["Zxy"]

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*.nc"))
    if len(file_list) == 0:
        raise ValueError("ERROR: No restart files found")

    copy_vars = [
        "BOUT_VERSION",
        "NXPE",
        "NYPE",
        "hist_hi",
        "tt",
        "MXG",
        "MYG",
        "MZG",
        "nz",
        "MZ",
        "run_id",
        "run_restart_from",
    ]
    copy_data = {}
    interp_vars = []

    # Read information from a restart file
    with DataFile(file_list[0]) as f:
        for var in copy_vars:
            try:
                copy_data[var] = f[var]
            except:
                print("Failed to copy variable '{}'".format(var))

        # Get a list of variables
        varnames = f.list()

        for var in varnames:
            dimensions = f.dimensions(var)
            if dimensions == ("x", "y", "z"):
                # Could be an evolving variable [x,y,z]
                interp_vars.append(var)

    # Only tested for nz = 1
    assert copy_data["nz"] == 1

    def extrapolate_yguards(data2d, myg):
        nx, ny = data2d.shape
        result = np.zeros((nx, ny + 2 * myg))
        print(result.shape)
        result[:, myg:-myg] = data2d

        dy = result[:, myg + 1] - result[:, myg]
        for i in range(1, myg + 1):
            result[:, myg - i] = result[:, myg] - i * dy
        dy = result[:, -myg - 1] - result[:, -myg - 2]
        for i in range(1, myg + 1):
            result[:, -myg - 1 + i] = result[:, -myg - 1] + i * dy
        return result

    from_Rxy = extrapolate_yguards(from_Rxy, copy_data["MYG"])
    from_Zxy = extrapolate_yguards(from_Zxy, copy_data["MYG"])
    to_Rxy = extrapolate_yguards(to_Rxy, copy_data["MYG"])
    to_Zxy = extrapolate_yguards(to_Zxy, copy_data["MYG"])

    interp_data = {}
    for var in interp_vars:
        print("Interpolating " + var)

        from_data = collect(
            var,
            path=path,
            xguards=True,
            yguards=True,
            prefix="BOUT.restart",
            info=False,
        )

        if interpolator == "CloughTocher":
            # Triangulate
            from scipy.interpolate import CloughTocher2DInterpolator

            interp = CloughTocher2DInterpolator(
                list(zip(from_Rxy.flatten(), from_Zxy.flatten())), from_data.flatten()
            )
        elif interpolator == "RBF":
            # Radial Basis Functions
            from scipy.interpolate import RBFInterpolator

            interp = RBFInterpolator(
                list(zip(from_Rxy.flatten(), from_Zxy.flatten())),
                from_data.flatten(),
                neighbors=50,
            )
        elif interpolator == "nearest":
            # Nearest neighbour. Tends to be robust
            from scipy.interpolate import NearestNDInterpolator

            interp = NearestNDInterpolator(
                list(zip(from_Rxy.flatten(), from_Zxy.flatten())), from_data.flatten()
            )
        elif interpolator == "linear":
            # Linear interpolator
            from scipy.interpolate import LinearNDInterpolator
            interp = LinearNDInterpolator(
                list(zip(from_Rxy.flatten(), from_Zxy.flatten())), from_data.flatten(),
                fill_value = 0.0
            )
        else:
            raise ValueError("Invalid interpolator")

        to_data = interp(list(zip(to_Rxy.flatten(), to_Zxy.flatten()))).reshape(
            to_Rxy.shape
        )

        print(
            "\tData ranges: {}:{} -> {}:{}".format(
                np.amin(from_data),
                np.amax(from_data),
                np.amin(to_data),
                np.amax(to_data),
            )
        )

        if floor:
            if var[0] == 'N' and var[1] != 'V':
                # A density
                to_data = np.clip(to_data, 1e-5, None)
                print("\t\t floor -> {}:{}".format(np.amin(to_data), np.amax(to_data)))
            elif var[0] == 'P':
                # Pressure
                to_data = np.clip(to_data, 1e-8, None)
                print("\t\t floor -> {}:{}".format(np.amin(to_data), np.amax(to_data)))

        if show:
            import matplotlib.pyplot as plt

            plt.pcolormesh(to_Rxy, to_Zxy, to_data, shading="auto")
            plt.plot(from_Rxy, from_Zxy, "ok")
            plt.colorbar()
            plt.axis("equal")
            plt.show()

        interp_data[var] = to_data

    # Now have copy_data and interp_data dictionaries to write to the
    # new restart files. Now need to partition the interpolated arrays
    # with similar logic to redistribute()

    nxpe = copy_data["NXPE"]
    nype = copy_data["NYPE"]
    npes = nxpe * nype

    mxg = copy_data["MXG"]
    myg = copy_data["MYG"]

    new_nx, new_ny = to_Rxy.shape

    if (new_nx - 2 * mxg) % nxpe != 0:
        # Can't split grid in this way
        raise ValueError("nxpe={} not compatible with nx = {}".format(nxpe, new_nx))
    if (new_ny - 2 * myg) % nxpe != 0:
        # Can't split grid in this way
        raise ValueError("nype={} not compatible with ny = {}".format(nype, new_ny))

    mxsub = (new_nx - 2 * mxg) // nxpe
    mysub = (new_ny - 2 * myg) // nype

    copy_data["MXSUB"] = mxsub
    copy_data["MYSUB"] = mysub
    copy_data["nx"] = new_nx
    copy_data["ny"] = new_ny - 2 * myg

    for i in range(npes):
        ix = i % nxpe
        iy = i // nxpe

        def get_block(data):
            sliced = data[
                ix * mxsub : (ix + 1) * mxsub + 2 * mxg,
                iy * mysub : (iy + 1) * mysub + 2 * myg,
            ]
            return sliced.reshape(sliced.shape + (1,))  # make 3D

        outpath = os.path.join(output, "BOUT.restart." + str(i) + ".nc")
        with DataFile(outpath, create=True) as f:
            print("Creating " + outpath)

            f.write("PE_XIND", ix)
            f.write("PE_YIND", iy)

            # Write the scalars
            for k in copy_data:
                f.write(k, copy_data[k])

            # Write fields
            for k in interp_data:
                f.write(k, get_block(interp_data[k]))


def shift_v3_to_v4(
    gridfile, zperiod, path="data", output=".", informat="nc", mxg=2, myg=2
):
    """Convert a set of restart files from BOUT++ v3 to v4

       Assumes that both simulations are using shifted metric coordinates:
       - v3 restarts are in field-aligned coordinates
         (shifted when taking X derivatives)
       - v4 restarts are in shifted coordinates
         (shifted when taking Y derivatives)

    Parameters
    ----------

    gridfile : str
        String containing grid file name
    zperiod : int
        Number of times the domain is repeated to form a full torus
    path : str, optional
        Directory containing the input restart files (BOUT++ v3)
    output : str, optional
        Directory where the output restart files will go
    informat : str, optional
        File extension of the input restart files
    mxg : int, optional
        Number of X guard cells
    myg : int, optional
        Number of Y guard cells
    """

    # Try opening the grid file
    with DataFile(gridfile) as grid:
        try:
            zShift = grid["zShift"]
        except KeyError:
            zShift = grid["qinty"]

    if path == output:
        raise ValueError("Can't overwrite restart file")

    file_list = glob.glob(os.path.join(path, "BOUT.restart.*." + informat))
    file_list = natsorted(file_list)
    nfiles = len(file_list)

    if nfiles == 0:
        raise ValueError("No data found")

    print("Number of files found: " + str(nfiles))

    # Read processor layout
    with DataFile(file_list[0]) as f:
        NXPE = f["NXPE"]
        NPES = f["NPES"]
    if NPES != nfiles:
        raise ValueError("Number of restart files doesn't match NPES")

    for n in range(nfiles):
        basename = "BOUT.restart.{}.{}".format(n, informat)
        f = os.path.join(path, basename)
        new_f = os.path.join(output, basename)
        print("Changing {} => {}".format(f, new_f))

        pe_xind = n % NXPE
        pe_yind = n // NXPE

        # Open the restart file in read mode and create the new file
        with DataFile(f) as old, DataFile(new_f, write=True, create=True) as new:
            # Add new variables
            new.write("MXG", mxg)
            new.write("MYG", myg)
            new.write("MZG", 0)
            new.write(
                "run_id",
                np.array(list("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"), dtype="c"),
            )
            new.write(
                "run_restart_from",
                np.array(list("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"), dtype="c"),
            )

            # Loop over the variables in the old file
            for var in old.list():
                # Read the data
                data = old.read(var)
                attributes = old.attributes(var)

                # Find 3D variables
                newdata = data
                if old.ndims(var) == 3:
                    print("    Shifting " + var)

                    # Remove one Z grid cell
                    newdata = newdata[:, :, :-1]

                    nx, ny, nz = newdata.shape
                    MXSUB = nx - 2 * mxg
                    MYSUB = ny - 2 * myg

                    x_offset = pe_xind * MXSUB
                    y_offset = pe_yind * MYSUB

                    # Shifting by zShift goes from field-aligned to orthogonal coords
                    # Note: Removing y guards but not X because zShift includes x boundaries
                    newdata[:, myg:-myg, :] = shiftz.shiftz(
                        newdata[:, myg:-myg, :],
                        zShift[
                            x_offset : (x_offset + nx), y_offset : (y_offset + MYSUB)
                        ],
                        zperiod=zperiod,
                    )
                    new.write("nx", nx)
                    new.write("ny", ny - 2 * myg)
                    new.write("nz", nz)
                newdata = BoutArray(newdata, attributes=attributes)
                new.write(var, newdata)
