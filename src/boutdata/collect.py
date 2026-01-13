import glob
import os
import pathlib
import sys

import numpy as np

from boututils.boutarray import BoutArray
from boututils.datafile import DataFile


def findVar(varname, varlist):
    """Find variable name in a list

    First does case insensitive comparison, then
    checks for abbreviations.

    Returns the matched string, or raises a ValueError

    Parameters
    ----------
    varname : str
        Variable name to look for
    varlist : list of str
        List of possible variable names

    Returns
    -------
    str
        The closest match to varname in varlist

    """
    # Try a variation on the case
    v = [name for name in varlist if name.lower() == varname.lower()]
    if len(v) == 1:
        # Found case match
        print(f"Variable '{varname}' not found. Using '{v[0]}' instead")
        return v[0]
    elif len(v) > 1:
        print(f"Variable '{varname}' not found, and is ambiguous. Could be one of: {v}")
        raise ValueError(f"Variable '{varname}' not found")

    # None found. Check if it's an abbreviation
    v = [name for name in varlist if name[: len(varname)].lower() == varname.lower()]
    if len(v) == 1:
        print(f"Variable '{varname}' not found. Using '{v[0]}' instead")
        return v[0]
    elif len(v) > 1:
        print(f"Variable '{varname}' not found, and is ambiguous. Could be one of: {v}")
    raise ValueError("Variable '" + varname + "' not found")


def _convert_to_nice_slice(r, N, name="range"):
    """Convert r to a "sensible" slice in range [0, N]

    If r is None, the slice corresponds to the full range.

    Lists or tuples of one or two ints are converted to slices.

    Slices with None for one or more arguments have them replaced with
    sensible values.

    Private helper function for collect

    Parameters
    ----------
    r : None, int, slice or list of int
        Range-like to check/convert to slice
    N : int
        Size of range
    name : str, optional
        Name of range for error message

    Returns
    -------
    slice
        "Sensible" slice with no Nones for start, stop or step
    """

    if N == 0:
        raise ValueError(f"No data available in {name}")
    if r is None:
        temp_slice = slice(N)
    elif isinstance(r, slice):
        temp_slice = r
    elif isinstance(r, int | np.integer):
        if r >= N or r < -N:
            # raise out of bounds error as if we'd tried to index the array with r
            # without this, would return an empty array instead
            raise IndexError(f"{name} index out of range, value was {r}")
        elif r == -1:
            temp_slice = slice(r, None)
        else:
            temp_slice = slice(r, r + 1)
    elif len(r) == 0:
        return _convert_to_nice_slice(None, N, name)
    elif len(r) == 1:
        return _convert_to_nice_slice(r[0], N, name)
    elif len(r) == 2:
        r2 = list(r)
        if r2[0] < 0:
            r2[0] = r2[0] + N
        if r2[1] < 0:
            r2[1] = r2[1] + N
        if r2[0] > r2[1]:
            raise ValueError("{} start ({}) is larger than end ({})".format(name, *r2))
        # Lists uses inclusive end, we need exclusive end
        temp_slice = slice(r2[0], r2[1] + 1)
    elif len(r) == 3:
        # Convert 3 element list to slice object
        temp_slice = slice(r[0], r[1], r[2])
    else:
        raise ValueError(f"Couldn't convert {name} ('{r}') to slice")

    # slice.indices converts None to actual values
    return slice(*temp_slice.indices(N))


def collect(
    varname,
    xind=None,
    yind=None,
    zind=None,
    tind=None,
    path=".",
    yguards=False,
    xguards=True,
    info=True,
    prefix="BOUT.dmp",
    strict=False,
    tind_auto=False,
    datafile_cache=None,
):
    """Collect a variable from a set of BOUT++ outputs.

    Parameters
    ----------
    varname : str
        Name of the variable
    xind, yind, zind, tind : int, slice or list of int, optional
        Range of X, Y, Z or time indices to collect. Either a single
        index to collect, a list containing [start, end] (inclusive
        end), or a slice object (usual python indexing). Default is to
        fetch all indices
    path : str, optional
        Path to data files (default: ".")
    prefix : str, optional
        File prefix (default: "BOUT.dmp")
    yguards : bool or "include_upper", optional
        Collect Y boundary guard cells? (default: False)
        If yguards=="include_upper" the y-boundary cells from the upper (second) target
        are also included.
    xguards : bool, optional
        Collect X boundary guard cells? (default: True)
        (Set to True to be consistent with the definition of nx)
    info : bool, optional
        Print information about collect? (default: True)
    strict : bool, optional
        Fail if the exact variable name is not found? (default: False)
    tind_auto : bool, optional
        Read all files, to get the shortest length of time_indices.
        Useful if writing got interrupted (default: False)
    datafile_cache : datafile_cache_tuple, optional
        Optional cache of open DataFile instances: namedtuple as returned
        by create_cache. Used by BoutOutputs to pass in a cache so that we
        do not have to re-open the dump files to read another variable
        (default: None)

    Examples
    --------

    >>> collect(name)
    BoutArray([[[[...]]]])

    """

    if datafile_cache is None:
        # Search for BOUT++ dump files
        file_list, parallel, _ = findFiles(path, prefix)
    else:
        parallel = datafile_cache.parallel
        file_list = datafile_cache.file_list

    def getDataFile(i):
        """Get the DataFile from the cache, if present, otherwise open the
        DataFile

        """
        if datafile_cache is not None:
            return datafile_cache.datafile_list[i]
        else:
            return DataFile(file_list[i])

    if parallel:
        return _collect_from_single_file(
            getDataFile(0),
            varname,
            xind,
            yind,
            zind,
            tind,
            path,
            yguards,
            xguards,
            info,
            prefix,
            strict,
            datafile_cache,
        )

    nfiles = len(file_list)

    # Read data from the first file
    f = getDataFile(0)
    grid_info, tind, xind, yind, zind = _get_grid_info(
        f,
        xguards=xguards,
        yguards=yguards,
        tind=tind,
        xind=xind,
        yind=yind,
        zind=zind,
        nfiles=len(file_list),
    )

    if varname not in grid_info["varNames"]:
        if strict:
            raise ValueError(f"Variable '{varname}' not found")
        else:
            varname = findVar(varname, f.list())

    dimensions = f.dimensions(varname)

    var_attributes = f.attributes(varname)
    ndims = len(dimensions)

    # ndims is 0 for reals, and 1 for f.ex. t_array
    if ndims == 0:
        # Just read from file
        data = f.read(varname)
        if datafile_cache is None:
            # close the DataFile if we are not keeping it in a cache
            f.close()
        return BoutArray(data, attributes=var_attributes)

    if ndims > 4:
        raise ValueError("ERROR: Too many dimensions")

    if tind_auto:
        nt = grid_info["nt"]
        for i in range(1, nfiles):
            f = getDataFile(i)
            t_array_ = f.read("t_array")
            nt = min(len(t_array_), nt)
            if datafile_cache is None:
                # close the DataFile if we are not keeping it in a cache
                f.close()
        grid_info["nt"] = nt

    if info:
        print(
            "mxsub = {} mysub = {} mz = {}\n".format(
                grid_info["mxsub"], grid_info["mysub"], grid_info["nz"]
            )
        )

        print(
            "nxpe = {}, nype = {}, npes = {}\n".format(
                grid_info["nxpe"], grid_info["nype"], grid_info["npes"]
            )
        )
        if grid_info["npes"] < nfiles:
            print("WARNING: More files than expected ({})".format(grid_info["npes"]))
        elif grid_info["npes"] > nfiles:
            print("WARNING: Some files missing. Expected {}".format(grid_info["npes"]))

    if not any(dim in dimensions for dim in ("x", "y", "z")):
        # Not a Field (i.e. no spatial dependence) so only read from the 0'th file
        result = _read_scalar(f, varname, dimensions, var_attributes, tind)
        if datafile_cache is None:
            # close the DataFile if we are not keeping it in a cache
            f.close()
        return result

    if datafile_cache is None:
        # close the DataFile if we are not keeping it in a cache
        f.close()

    # Create a list with size of each dimension
    ddims = [grid_info["sizes"][d] for d in dimensions]

    # Create the data array
    data = np.zeros(ddims)

    if dimensions == ("t", "x", "z") or dimensions == ("x", "z"):
        is_fieldperp = True
        yindex_global = None
        # The pe_yind that this FieldPerp is going to be read from
        fieldperp_yproc = None
    else:
        is_fieldperp = False

    for i in range(grid_info["npes"]):
        f = getDataFile(i)
        temp_yindex, temp_f_attributes = _collect_from_one_proc(
            i,
            f,
            varname,
            result=data,
            is_fieldperp=is_fieldperp,
            dimensions=dimensions,
            grid_info=grid_info,
            tind=tind,
            xind=xind,
            yind=yind,
            zind=zind,
            xguards=xguards,
            yguards=(yguards is not False),
            info=info,
        )
        if is_fieldperp:
            (
                yindex_global,
                fieldperp_yproc,
                var_attributes,
            ) = _check_fieldperp_attributes(
                varname,
                yindex_global,
                temp_yindex,
                i // grid_info["nxpe"],
                fieldperp_yproc,
                var_attributes,
                temp_f_attributes,
            )
        if datafile_cache is None:
            # close the DataFile if we are not keeping it in a cache
            f.close()

    # if a step was requested in x or y, need to apply it here
    data = _apply_step(data, dimensions, xind.step, yind.step)

    # Finished looping over all files
    if info:
        sys.stdout.write("\n")
    return BoutArray(data, attributes=var_attributes)


def _collect_from_single_file(
    f,
    varname,
    xind,
    yind,
    zind,
    tind,
    path,
    yguards,
    xguards,
    info,
    prefix,
    strict,
    datafile_cache,
):
    """
    Collect data from a single file

    Single file may be created by parallel writing saving all BOUT++ output to a single
    file, or by squashoutput() 'squashing' data from one file per processor into a
    single file.

    Parameters
    ----------
    f : DataFile
        Single file to read data from
    For description of remaining arguments, see docstring of collect().
    """
    if info:
        print("Single (parallel) data file")

    if varname not in f.keys():
        if strict:
            raise ValueError(f"Variable '{varname}' not found")
        else:
            varname = findVar(varname, f.list())

    dimensions = f.dimensions(varname)

    try:
        mxg = f["MXG"]
    except KeyError:
        mxg = 0
        print(f"MXG not found, setting to {mxg}")
    try:
        myg = f["MYG"]
    except KeyError:
        myg = 0
        print(f"MYG not found, setting to {myg}")

    if xguards:
        nx = f["nx"]
    else:
        nx = f["nx"] - 2 * mxg
    if yguards:
        ny = f["ny"] + 2 * myg
        if yguards == "include_upper" and f["jyseps2_1"] != f["jyseps1_2"]:
            # Simulation has a second (upper) target, with a second set of y-boundary
            # points
            ny = ny + 2 * myg
    else:
        ny = f["ny"]
    nz = f["MZ"]
    t_array = f.read("t_array")
    if t_array is None:
        nt = 1
        t_array = np.zeros(1)
    else:
        try:
            nt = len(t_array)
        except TypeError:
            # t_array is not an array here, which probably means it was a
            # one-element array and has been read as a scalar.
            nt = 1

    xind = _convert_to_nice_slice(xind, nx, "xind")
    yind = _convert_to_nice_slice(yind, ny, "yind")
    zind = _convert_to_nice_slice(zind, nz, "zind")
    tind = _convert_to_nice_slice(tind, nt, "tind")

    if not xguards:
        xind = slice(xind.start + mxg, xind.stop + mxg, xind.step)
    if not yguards:
        yind = slice(yind.start + myg, yind.stop + myg, yind.step)

    dim_ranges = {"t": tind, "x": xind, "y": yind, "z": zind}
    ranges = [dim_ranges.get(dim, None) for dim in dimensions]

    data = f.read(varname, ranges)
    var_attributes = f.attributes(varname)
    return BoutArray(data, attributes=var_attributes)


def _read_scalar(f, varname, dimensions, var_attributes, tind):
    """
    Read a scalar variable from a single file

    Parameters
    ----------
    f : DataFile
        File to read from. This function does *not* close f.
    varname : str
        Name of variable to read
    dimensions : tuple
        Dimensions of the variable
    var_attributes : dict
        Attributes of the variable
    tind : slice
        Slice to apply to the t-dimension, if there is one
    """
    if "t" in dimensions:
        if not dimensions[0] == "t":
            # 't' should be the first dimension in the list if present
            raise ValueError(
                f"{varname} has a 't' dimension, but it is not the first dimension "
                f"in dimensions={dimensions}"
            )
        data = f.read(varname, ranges=[tind] + (len(dimensions) - 1) * [None])
    else:
        # No time or space dimensions, so no slicing
        data = f.read(varname)
    return BoutArray(data, attributes=var_attributes)


def _apply_step(data, dimensions, xstep, ystep):
    """
    Apply steps of xind and yind slices to an array

    Parameters
    ----------
    data : np.Array
        Data array to be sliced
    dimensions : tuple
        Dimensions of data
    xstep : int or None
        Step to apply in the x-direction
    ystep : int or None
        Step to apply in the y-direction
    """
    slices = [slice(None)] * len(dimensions)

    if "x" in dimensions:
        slices[dimensions.index("x")] = slice(None, None, xstep)

    if "y" in dimensions:
        slices[dimensions.index("y")] = slice(None, None, ystep)

    return data[tuple(slices)]


def _collect_from_one_proc(
    i,
    datafile,
    varname,
    *,
    result,
    is_fieldperp,
    grid_info,
    dimensions,
    tind,
    xind,
    yind,
    zind,
    xguards,
    yguards,
    info,
    parallel_read=False,
):
    """Read part of a variable from one processor

    Reads the part of the data from the file output by a single processor. Excludes
    guard cells used only for communication between processors, but optionally includes
    boundary cells.

    Result is stored into the global array passed in the 'result'
    argument - this avoids complicated concatenation of results from multiple
    processors, and is also more convenient when using a shared memory array to gather
    results from parallel workers.

    The returned values are used for checks on FieldPerp variables.

    Parameters
    ----------
    i : int
        Processor number being read from
    datafile : DataFile
        File to read from
    varname : str
        Name of variable to read
    result : numpy.Array
        Array in which to put the data
    is_fieldperp : bool
        Is this variable a FieldPerp?
    grid_info : dict
        dict of grid parameters
    dimensions : tuple of str
        Dimensions of the variable
    tind : slice
        Slice for t-dimension
    xind : slice
        Slice for x-dimension
    yind : slice
        Slice for y-dimension
    zind : slice
        Slice for z-dimension
    xguards : bool
        Include x-boundary cells at either side of the global grid?
    yguards : bool
        Include y-boundary cells at either end of the global grid?

    Returns
    -------
    temp_yindex, var_attributes
    """
    ndims = len(dimensions)

    # ndims is 0 for reals, and 1 for f.ex. t_array
    if ndims == 0:
        if i != 0:
            # Only read scalars from file 0
            return None, None

        # Just read from file
        result[...] = datafile.read(varname)
        return None, None

    if ndims > 4:
        raise ValueError("ERROR: Too many dimensions")

    if not any(dim in dimensions for dim in ("x", "y", "z")):
        if i != 0:
            return None, None

        # Not a Field (i.e. no spatial dependence) so only read from the 0'th file
        if "t" in dimensions:
            if not dimensions[0] == "t":
                # 't' should be the first dimension in the list if present
                raise ValueError(
                    f"{varname} has a 't' dimension, but it is not the first dimension "
                    f"in dimensions={dimensions}"
                )
            result[:] = datafile.read(varname, ranges=[tind] + (ndims - 1) * [None])
        else:
            # No time or space dimensions, so no slicing
            result[...] = datafile.read(varname)
        return None, None

    nxpe = grid_info["nxpe"]
    nype = grid_info["nype"]
    mxsub = grid_info["mxsub"]
    mysub = grid_info["mysub"]
    mxg = grid_info["mxg"]
    myg = grid_info["myg"]
    yproc_upper_target = grid_info["yproc_upper_target"]

    # Get X and Y processor indices
    pe_yind = i // nxpe
    pe_xind = i % nxpe

    inrange = True

    xstart, xstop, xgstart, xgstop, inrange = _get_x_range(
        xguards, xind, pe_xind, nxpe, mxsub, mxg, inrange
    )
    if is_fieldperp:
        # FieldPerps do not have a y-dimension, so cannot be sliced in y and should
        # always be read regardless of the value of yind (so we should not change
        # inrange by checking the y-range).
        # ystart, ystop, ygstart and ygstop are set only to avoid errors in 'info'
        # messages.
        ystart = 0
        ystop = 1
        ygstart = 0
        ygstop = 1
    else:
        ystart, ystop, ygstart, ygstop, inrange = _get_y_range(
            yguards, yind, pe_yind, nype, yproc_upper_target, mysub, myg, inrange
        )

    if not inrange:
        return None, None  # Don't need this file

    local_dim_slices = {
        "t": tind,
        "x": slice(xstart, xstop),
        "y": slice(ystart, ystop),
        "z": zind,
    }
    local_slices = tuple(local_dim_slices.get(dim, None) for dim in dimensions)

    global_dim_slices = {"x": slice(xgstart, xgstop), "y": slice(ygstart, ygstop)}
    if parallel_read:
        # When reading in parallel, we are always reading into a 4-dimensional shared
        # array.  Should not reach this function unless we only have dimensions in
        # ("t", "x", "y", "z")
        global_slices = tuple(
            global_dim_slices.get(dim, slice(None)) if dim in dimensions else 0
            for dim in ("t", "x", "y", "z")
        )
    else:
        # Otherwise, reading into an array with the same dimensions as the variable.
        global_slices = tuple(
            global_dim_slices.get(dim, slice(None)) for dim in dimensions
        )

    if info:
        print(
            f"\rReading from {i}: [{xstart}-{xstop - 1}][{ystart}-{ystop - 1}] -> [{xgstart}-{xgstop - 1}][{ygstart}-{ygstop - 1}]\n"
        )

    if is_fieldperp:
        f_attributes = datafile.attributes(varname)
        temp_yindex = f_attributes["yindex_global"]
        if temp_yindex < 0 or not _fieldperp_from_this(
            nype, pe_yind, mysub, myg, temp_yindex
        ):
            # No data for FieldPerp on this processor
            return None, None

    result[global_slices] = datafile.read(varname, ranges=local_slices)

    if is_fieldperp:
        return temp_yindex, f_attributes

    return None, None


def _fieldperp_from_this(nype, pe_yind, mysub, myg, temp_yindex):
    if pe_yind == 0:
        if temp_yindex < mysub + myg:
            return True
    if pe_yind == nype - 1:
        if myg + mysub * pe_yind <= temp_yindex:
            return True
    return myg + mysub * pe_yind <= temp_yindex < myg + mysub * (pe_yind + 1)


def _check_local_range_lower(start, stop, lower_index, inrange):
    """
    Utility function for _get_x_range and _get_y_range. Checks inner or lower edge of
    local ranges.

    Parameters
    ----------
    start : int
        Initial version of local index where slice starts. Reset to lower_index if
        smaller than lower_index.
    stop : int
        Local index where slice stops.
    lower_index : int
        Local index where valid data (including boundaries if necessary) starts on
        current processor.
    inrange : bool
        Initial value of inrange, which is True if data on current processor is within
        the global range requested. Updated if stop is less than or equal to
        lower_index.

    Returns
    -------
    start : int
        Updated (if necessary) version of start argument
    inrange : bool
        Updated (if necessary) version of inrange argument
    """
    if start < lower_index:
        start = lower_index
    if stop <= lower_index:
        inrange = False
    return start, inrange


def _check_local_range_upper(start, stop, upper_index, inrange):
    """
    Utility function for _get_x_range and _get_y_range. Checks outer or upper edge of
    local ranges.

    Parameters
    ----------
    start : int
        Local index where slice starts.
    stop : int
        Initial version of local index where slice stops. Reset to upper_index if
        larger than upper_index.
    upper_index : int
        Local index where valid data (including boundaries if necessary) stops on
        current processor.
    inrange : bool
        Initial value of inrange, which is True if data on current processor is within
        the global range requested. Updated if start is greater than or equal to
        upper_index.

    Returns
    -------
    stop : int
        Updated (if necessary) version of stop argument
    inrange : bool
        Updated (if necessary) version of inrange argument
    """
    if start >= upper_index:
        inrange = False
    if stop > upper_index:
        stop = upper_index
    return stop, inrange


def _get_x_range(xguards, xind, pe_xind, nxpe, mxsub, mxg, inrange):
    """
    Get local ranges of x-indices

    Parameters
    ----------
    xguards : bool
        Include x-boundaries?
    xind : slice
        Global slice to apply to x-dimension
    pe_xind : int
        x-index of the processor
    nxpe : int
        Number of processors in the x-direction
    mxsub : int
        Number of grid cells (excluding guard cells) in the x-direction on a single
        procssor
    mxg : int
        Number of guard cells in the x-direction
    inrange : bool
        Does the processor have data to read?

    Returns
    -------
    xstart : int
        Local x-index to start reading
    xstop : int
        Local x-index to stop reading
    xgstart : int
        Global x-index to start putting data
    xgstop : int
        Global x-index to stop putting data
    inrange : bool
        Updated version of inrange - changed to False if this processor has no data to
        read
    """
    # Local ranges
    if xguards:
        xstart = xind.start - pe_xind * mxsub
        xstop = xind.stop - pe_xind * mxsub

        # Check lower x boundary
        if pe_xind == 0:
            # Keeping inner boundary
            xstart, inrange = _check_local_range_lower(xstart, xstop, 0, inrange)
        else:
            xstart, inrange = _check_local_range_lower(xstart, xstop, mxg, inrange)

        # Upper x boundary
        if pe_xind == (nxpe - 1):
            # Keeping outer boundary
            xstop, inrange = _check_local_range_upper(
                xstart, xstop, mxsub + 2 * mxg, inrange
            )
        else:
            xstop, inrange = _check_local_range_upper(
                xstart, xstop, mxsub + mxg, inrange
            )

    else:
        xstart = xind.start - pe_xind * mxsub + mxg
        xstop = xind.stop - pe_xind * mxsub + mxg

        xstart, inrange = _check_local_range_lower(xstart, xstop, mxg, inrange)
        xstop, inrange = _check_local_range_upper(xstart, xstop, mxsub + mxg, inrange)

    # Global ranges
    if xguards:
        xgstart = xstart + pe_xind * mxsub - xind.start
        xgstop = xstop + pe_xind * mxsub - xind.start
    else:
        xgstart = xstart + pe_xind * mxsub - mxg - xind.start
        xgstop = xstop + pe_xind * mxsub - mxg - xind.start

    return xstart, xstop, xgstart, xgstop, inrange


def _get_y_range(yguards, yind, pe_yind, nype, yproc_upper_target, mysub, myg, inrange):
    """
    Get local ranges of y-indices

    Parameters
    ----------
    yguards : bool
        Include y-boundaries?
    yind : slice
        Global slice to apply to y-dimension
    pe_yind : int
        y-index of the processor
    nype : int
        Number of processors in the y-direction
    yproc_upper_target : int or None
        Index of processor whose lower y-boundary is the upper target, if there is an
        upper target
    mysub : int
        Number of grid cells (excluding guard cells) in the y-direction on a single
        procssor
    myg : int
        Number of guard cells in the y-direction
    inrange : bool
        Does the processor have data to read?

    Returns
    -------
    ystart : int
        Local y-index to start reading
    ystop : int
        Local y-index to stop reading
    ygstart : int
        Global y-index to start putting data
    ygstop : int
        Global y-index to stop putting data
    inrange : bool
        Updated version of inrange - changed to False if this processor has no data to
        read
    """
    # Local ranges
    if yguards:
        ystart = yind.start - pe_yind * mysub
        ystop = yind.stop - pe_yind * mysub

        # Check lower y boundary
        if pe_yind == 0:
            # Keeping inner boundary
            ystart, inrange = _check_local_range_lower(ystart, ystop, 0, inrange)
        else:
            ystart, inrange = _check_local_range_lower(ystart, ystop, myg, inrange)
        # and lower y boundary at upper target
        if yproc_upper_target is not None and pe_yind - 1 == yproc_upper_target:
            ystart = ystart - myg

        # Upper y boundary
        if pe_yind == (nype - 1):
            # Keeping outer boundary
            ystop, inrange = _check_local_range_upper(
                ystart, ystop, mysub + 2 * myg, inrange
            )
        else:
            ystop, inrange = _check_local_range_upper(
                ystart, ystop, mysub + myg, inrange
            )
        # upper y boundary at upper target
        if yproc_upper_target is not None and pe_yind == yproc_upper_target:
            ystop = ystop + myg

    else:
        ystart = yind.start - pe_yind * mysub + myg
        ystop = yind.stop - pe_yind * mysub + myg

        ystart, inrange = _check_local_range_lower(ystart, ystop, myg, inrange)
        ystop, inrange = _check_local_range_upper(ystart, ystop, mysub + myg, inrange)

    # Global ranges
    if yguards:
        ygstart = ystart + pe_yind * mysub - yind.start
        ygstop = ystop + pe_yind * mysub - yind.start
        if yproc_upper_target is not None and pe_yind > yproc_upper_target:
            ygstart = ygstart + 2 * myg
            ygstop = ygstop + 2 * myg
    else:
        ygstart = ystart + pe_yind * mysub - myg - yind.start
        ygstop = ystop + pe_yind * mysub - myg - yind.start

    return ystart, ystop, ygstart, ygstop, inrange


def _check_fieldperp_attributes(
    varname,
    yindex_global,
    temp_yindex,
    pe_yind,
    fieldperp_yproc,
    var_attributes,
    temp_f_attributes,
):
    """
    Check attributes for a FieldPerp from one file. If the FieldPerp was actually
    written to that file, update the 'global' attributes of the FieldPerp. If data for
    the FieldPerp has already been found, check that the y-index of the processors is
    the same and the 'yindex_global' is the same.
    """
    if temp_yindex is not None:
        # Found actual data for a FieldPerp, so update FieldPerp properties
        # and check they are unique
        if yindex_global is not None and yindex_global != temp_yindex:
            raise ValueError(
                f"Found FieldPerp {varname} at different global y-indices, {temp_yindex} and {yindex_global}"
            )
        yindex_global = temp_yindex
        if fieldperp_yproc is not None and fieldperp_yproc != pe_yind:
            raise ValueError(
                f"Found FieldPerp {varname} on different y-processor indices, {fieldperp_yproc} and {pe_yind}"
            )
        fieldperp_yproc = pe_yind
        var_attributes = temp_f_attributes

    return yindex_global, fieldperp_yproc, var_attributes


def _get_grid_info(
    f, *, xguards, yguards, tind, xind, yind, zind, nfiles, all_vars_info=False
):
    """Get the grid info from an open DataFile

    Parameters
    ----------
    f : DataFile
        File to read grid info from
    xguards : bool
        Keeping x boundaries?
    yguards : bool or "include_upper"
        Keeping y boundaries?
    tind : int, sequence of int or slice
        Slice for t-dimension
    xind : int, sequence of int or slice
        Slice for x-dimension
    yind : int, sequence of int or slice
        Slice for y-dimension
    zind : int, sequence of int or slice
        Slice for z-dimension
    nfiles : int
        Number of files being read from
    all_vars_info : bool, default False
        Load extra info on names, dimensions and attributes of all variables.
    """

    def load_and_check(varname):
        var = f.read(varname)
        if var is None:
            raise ValueError(f"Missing {varname} variable")
        return var

    mz = int(load_and_check("MZ"))

    # Get the version of BOUT++ (should be > 0.6 for NetCDF anyway)
    try:
        version = f["BOUT_VERSION"]
    except KeyError:
        print("BOUT++ version : Pre-0.2")
        version = 0

    mxg = int(load_and_check("MXG"))
    myg = int(load_and_check("MYG"))
    mxsub = int(load_and_check("MXSUB"))
    mysub = int(load_and_check("MYSUB"))
    try:
        nxpe = int(f["NXPE"])
    except KeyError:
        nxpe = 1
        print(f"NXPE not found, setting to {nxpe}")
    try:
        nype = int(f["NYPE"])
    except KeyError:
        nype = nfiles
        print(f"NYPE not found, setting to {nype}")

    if "t_array" in f.keys():
        nt = len(f.read("t_array"))
    else:
        nt = 1
    nx = nxpe * mxsub + 2 * mxg if xguards else nxpe * mxsub

    yproc_upper_target = None
    if yguards:
        ny = mysub * nype + 2 * myg
        if yguards == "include_upper":
            is_doublenull = load_and_check("jyseps2_1") != load_and_check("jyseps1_2")
            if is_doublenull:
                ny_inner = int(load_and_check("ny_inner"))
                # Simulation has a second (upper) target, with a second set of
                # y-boundary points
                ny = ny + 2 * myg
                yproc_upper_target = ny_inner // mysub - 1
                if ny_inner % mysub != 0:
                    raise ValueError(
                        f"Trying to keep upper boundary cells but mysub={mysub} does not "
                        f"divide ny_inner={ny_inner}"
                    )
    else:
        ny = mysub * nype

    nz = mz - 1 if version < 3.5 else mz

    tind = _convert_to_nice_slice(tind, nt, "tind")
    xind = _convert_to_nice_slice(xind, nx, "xind")
    yind = _convert_to_nice_slice(yind, ny, "yind")
    zind = _convert_to_nice_slice(zind, nz, "zind")

    xsize = xind.stop - xind.start
    ysize = yind.stop - yind.start
    zsize = int(np.ceil(float(zind.stop - zind.start) / zind.step))
    tsize = int(np.ceil(float(tind.stop - tind.start) / tind.step))

    # Map between dimension names and output size
    sizes = {"x": xsize, "y": ysize, "z": zsize, "t": tsize}

    varNames = f.keys()

    result = {
        "mxg": mxg,
        "mxsub": mxsub,
        "myg": myg,
        "mysub": mysub,
        "nt": nt,
        "npes": nxpe * nype,
        "nx": nx,
        "nxpe": nxpe,
        "ny": ny,
        "nype": nype,
        "nz": nz,
        "sizes": sizes,
        "varNames": varNames,
        "yproc_upper_target": yproc_upper_target,
    }

    if yguards == "include_upper":
        result["is_doublenull"] = is_doublenull
        if is_doublenull:
            result["ny_inner"] = ny_inner

    if all_vars_info:
        attributes = {}
        dimensions = {}
        evolvingVariableNames = []
        for name in varNames:
            attributes[name] = f.attributes(name)
            var_dimensions = f.dimensions(name)
            dimensions[name] = var_dimensions
            if name != "t_array" and "t" in var_dimensions:
                evolvingVariableNames.append(name)
        result["attributes"] = attributes
        result["dimensions"] = dimensions
        result["evolvingVariableNames"] = evolvingVariableNames

    return result, tind, xind, yind, zind


def attributes(varname, path=".", prefix="BOUT.dmp"):
    """Return a dictionary of variable attributes in an output file

    Parameters
    ----------
    varname : str
        Name of the variable
    path : str, optional
        Path to data files (default: ".")
    prefix : str, optional
        File prefix (default: "BOUT.dmp")

    Returns
    -------
    dict
        A dictionary of attributes of varname
    """
    # Search for BOUT++ dump files in NetCDF format
    file_list, _, _ = findFiles(path, prefix)

    # Read data from the first file
    f = DataFile(file_list[0])

    return f.attributes(varname)


def dimensions(varname, path=".", prefix="BOUT.dmp"):
    """Return the names of dimensions of a variable in an output file

    Parameters
    ----------
    varname : str
        Name of the variable
    path : str, optional
        Path to data files (default: ".")
    prefix : str, optional
        File prefix (default: "BOUT.dmp")

    Returns
    -------
    tuple of strs
        The elements of the tuple give the names of corresponding variable
        dimensions

    """
    file_list, _, _ = findFiles(path, prefix)
    return DataFile(file_list[0]).dimensions(varname)


def findFiles(path, prefix):
    """Find files matching prefix in path.

    Netcdf (".nc", ".ncdf", ".cdl") and HDF5 (".h5", ".hdf5", ".hdf")
    files are searched.

    Parameters
    ----------
    path : str
        Path to data files
    prefix : str
        File prefix

    Returns
    -------
    tuple : (list of str, bool, str)
        The first element of the tuple is the list of files, the second is
        whether the files are a parallel dump file and the last element is
        the file suffix.

    """

    # Make sure prefix does not have a trailing .
    if prefix[-1] == ".":
        prefix = prefix[:-1]

    # Look for parallel dump files
    suffixes = [".nc", ".ncdf", ".cdl", ".h5", ".hdf5", ".hdf", ".bp"]
    file_list_parallel = None
    suffix_parallel = ""
    for test_suffix in suffixes:
        files = glob.glob(os.path.join(path, prefix + test_suffix))
        if files:
            if file_list_parallel:  # Already had a list of files
                raise OSError(
                    f"Parallel dump files with both {suffix_parallel} and {test_suffix} extensions are present. "
                    "Do not know which to read."
                )
            suffix_parallel = test_suffix
            file_list_parallel = files

    file_list = None
    suffix = ""
    for test_suffix in suffixes:
        files = glob.glob(os.path.join(path, prefix + ".*" + test_suffix))
        if files:
            if file_list:  # Already had a list of files
                raise OSError(
                    f"Dump files with both {suffix} and {test_suffix} extensions are present. Do not "
                    "know which to read."
                )
            suffix = test_suffix
            file_list = files

    if file_list_parallel and file_list:
        raise OSError(
            f"Both regular (with suffix {suffix}) and parallel (with suffix {suffix_parallel}) dump files "
            "are present. Do not know which to read."
        )
    elif file_list_parallel:
        return file_list_parallel, True, suffix_parallel
    elif file_list:
        path = pathlib.Path(path)
        # make sure files are in the right order
        nfiles = len(file_list)
        file_list = [path / f"{prefix}.{i}{suffix}" for i in range(nfiles)]
        return file_list, False, suffix
    else:
        raise OSError(f"ERROR: No data files found in path {path}")


def create_cache(path, prefix):
    """Create a list of DataFile objects to be passed repeatedly to
    collect.

    Parameters
    ----------
    path : str
        Path to data files
    prefix : str
        File prefix

    Returns
    -------
    namedtuple : (list of str, bool, str,
                  list of :py:obj:`~boututils.datafile.DataFile`)
        The cache of DataFiles in a namedtuple along with the file_list,
        and parallel and suffix attributes

    """

    # define namedtuple to return as the result
    from collections import namedtuple

    datafile_cache_tuple = namedtuple(
        "datafile_cache", ["file_list", "parallel", "suffix", "datafile_list"]
    )

    file_list, parallel, suffix = findFiles(path, prefix)

    cache = []
    for f in file_list:
        cache.append(DataFile(f))

    return datafile_cache_tuple(
        file_list=file_list, parallel=parallel, suffix=suffix, datafile_list=cache
    )
