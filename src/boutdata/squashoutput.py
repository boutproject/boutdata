# PYTHON_ARGCOMPLETE_OK

"""
Collect all data from BOUT.dmp.* files and create a single output file.

Output file named BOUT.dmp.nc by default

Useful because this discards ghost cell data (that is only useful for debugging)
and because single files are quicker to download.

"""

from . import __version__


def squashoutput(
    datadir=".",
    outputname="BOUT.dmp.nc",
    format="NETCDF4",
    tind=None,
    xind=None,
    yind=None,
    zind=None,
    xguards=True,
    yguards="include_upper",
    zguards=True,
    drop_variables=None,
    singleprecision=False,
    compress=False,
    least_significant_digit=None,
    quiet=False,
    complevel=None,
    append=False,
    delete=False,
    tind_auto=False,
    parallel=False,
    time_split_size=None,
    time_split_first_label=0,
):
    """
    Collect all data from BOUT.dmp.* files and create a single output file.

    Note: adds an attribute to the 'squashed' output file called `squashoutput_version`
    which records the current version number of `boutdata`.

    Parameters
    ----------
    datadir : str
        Directory where dump files are and where output file will be created.
        default "."
    outputname : str
        Name of the output file. File suffix specifies whether to use NetCDF or
        HDF5 (see boututils.datafile.DataFile for suffixes).
        default "BOUT.dmp.nc"
    format : str
        format argument passed to DataFile
        default "NETCDF4"
    tind : slice, int, or [int, int, int]
        tind argument passed to collect
        default None
    xind : slice, int, or [int, int, int]
        xind argument passed to collect
        default None
    yind : slice, int, or [int, int, int]
        yind argument passed to collect
        default None
    zind : slice, int, or [int, int, int]
        zind argument passed to collect
        default None
    xguards : bool
        xguards argument passed to collect
        default True
    yguards : bool or "include_upper"
        yguards argument passed to collect (note different default to collect's)
        default "include_upper"
    drop_variables : str, or list or tuple of strings
        Variable names passed in drop_variables will be ignored, and not included in the
        squashed output file.
    singleprecision : bool
        If true convert data to single-precision floats
        default False
    compress : bool
        If true enable compression in the output file
    least_significant_digit : int or None
        How many digits should be retained? Enables lossy
        compression. Default is lossless compression. Needs
        compression to be enabled.
    complevel : int or None
        Compression level, 1 should be fastest, and 9 should yield
        highest compression.
    quiet : bool
        Be less verbose. default False
    append : bool
        Append to existing squashed file
    delete : bool
        Delete the original files after squashing.
    tind_auto : bool, optional
        Read all files, to get the shortest length of time_indices. All data truncated
        to the shortest length.  Useful if writing got interrupted (default: False)
    parallel : bool or int, default False
        If set to True or 0, use the multiprocessing library to read data in parallel
        with the maximum number of available processors. If set to an int, use that many
        processes.
    time_split_size : int, optional
        By default no splitting is done. If an integer value is passed, the output is
        split into files with length in the t-dimension equal to that value. The outputs
        are labelled by prefacing a counter (starting by default at 0, but see
        time_split_first_label) to the file name before the .nc suffix.
    time_split_first_label : int, default 0
        Value at which to start the counter labelling output files when time_split_size
        is used.
    """
    # use local imports to allow fast import for tab-completion
    import glob
    import os
    import shutil
    import tempfile

    import numpy

    from boutdata.data import BoutOutputs
    from boututils.boutarray import BoutArray
    from boututils.datafile import DataFile

    try:
        # If we are using the netCDF4 module (the usual case) set caching to zero, since
        # each variable is read and written exactly once so caching does not help, only
        # uses memory - for large data sets, the memory usage may become excessive.
        from netCDF4 import get_chunk_cache, set_chunk_cache
    except ImportError:
        netcdf4_chunk_cache = None
    else:
        netcdf4_chunk_cache = get_chunk_cache()
        set_chunk_cache(0)

    fullpath = os.path.join(datadir, outputname)

    if append:
        if time_split_size is not None:
            raise ValueError("'time_split_size' is not compatible with append=True")
        datadirnew = tempfile.mkdtemp(dir=datadir)
        for f in glob.glob(os.path.join(datadir, "BOUT.dmp.*.??")):
            if not quiet:
                print("moving", f, flush=True)
            shutil.move(f, datadirnew)
        oldfile = datadirnew + "/" + outputname
        datadir = datadirnew

    if not drop_variables:
        drop_variables = []
    elif isinstance(drop_variables, str):
        drop_variables = [drop_variables]

    # useful object from BOUT pylib to access output data
    outputs = BoutOutputs(
        datadir,
        info=False,
        xguards=xguards,
        yguards=yguards,
        zguards=zguards,
        tind=tind,
        xind=xind,
        yind=yind,
        zind=zind,
        tind_auto=tind_auto,
        parallel=parallel,
    )

    # Create file(s) for output and write data
    filenames, t_slices = _get_filenames_t_slices(
        time_split_size, time_split_first_label, fullpath, outputs.tind
    )

    if not append:
        for f in filenames:
            if os.path.isfile(f):
                raise ValueError(
                    f"{fullpath} already exists, squashoutput() will not overwrite. Also, "
                    "for some filenames collect may try to read from this file, which "
                    "is presumably not desired behaviour."
                )

    outputvars = [k for k in outputs.keys() if k not in drop_variables]

    # Read a value to cache the files
    outputs[outputvars[0]]

    if append:
        # move only after the file list is cached
        shutil.move(fullpath, oldfile)

    t_array_index = outputvars.index("t_array")
    outputvars.append(outputvars.pop(t_array_index))

    kwargs = {}
    if compress:
        kwargs["zlib"] = True
        if least_significant_digit is not None:
            kwargs["least_significant_digit"] = least_significant_digit
        if complevel is not None:
            kwargs["complevel"] = complevel
    if append:
        old = DataFile(oldfile)
        # Check if dump on restart was enabled
        # If so, we want to drop the duplicated entry
        cropnew = 0
        if old["t_array"][-1] == outputs["t_array"][0]:
            cropnew = 1
        # Make sure we don't end up with duplicated data:
        for ot in old["t_array"]:
            if ot in outputs["t_array"][cropnew:]:
                raise RuntimeError(
                    "For some reason t_array has some duplicated entries in the new "
                    "and old file."
                )
    kwargs["format"] = format

    files = [DataFile(name, create=True, write=True, **kwargs) for name in filenames]

    for varname in outputvars:
        if not quiet:
            print(varname, flush=True)

        var = outputs[varname]
        dims = outputs.dimensions[varname]
        if append:
            if "t" in dims:
                var = var[cropnew:, ...]
                varold = old[varname]
                var = BoutArray(numpy.append(varold, var, axis=0), var.attributes)

        if singleprecision:
            if not isinstance(var, int):
                var = BoutArray(numpy.float32(var), var.attributes)

        if "t" in dims:
            for f, t_slice in zip(files, t_slices):
                f.write(varname, var[t_slice])
        else:
            for f in files:
                f.write(varname, var)

        var = None

    # Copy file attributes
    for attrname in outputs.list_file_attributes():
        attrval = outputs.get_file_attribute(attrname)
        for f in files:
            f.write_file_attribute(attrname, attrval)

    f.write_file_attribute("squashoutput_version", __version__)

    for f in files:
        f.close()

    del outputs

    if delete:
        if append:
            os.remove(oldfile)
        for f in glob.glob(datadir + "/BOUT.dmp.*.??"):
            if not quiet:
                print("Deleting", f, flush=True)
            os.remove(f)
        if append:
            os.rmdir(datadir)

    if netcdf4_chunk_cache is not None:
        # Reset the default chunk_cache size that was changed for squashoutput
        # Note that get_chunk_cache() returns a tuple, so we have to unpack it when
        # passing to set_chunk_cache.
        set_chunk_cache(*netcdf4_chunk_cache)


def _get_filenames_t_slices(time_split_size, time_split_first_label, fullpath, tind):
    """
    Create the filenames and slices used for splitting output in time. If not
    splitting, 'do nothing'.

    Parameters
    ----------
    time_split_size : int or None
        See docstring of squashoutput().
    time_split_first_label : int
        See docstring of squashoutput().
    fullpath : str
        Path of the directory where data files are.
    tind : slice
        slice object applied to time dimension when reading data. Used to
        calculate length of time dimension when time_split_size is set.

    Returns
    -------
    filenames : list of str
        File names to write output to.
    t_slices : list of slice
        Slices to be applied to the time dimension to select data for each
        output file.
    """
    if time_split_size is None:
        return [fullpath], [slice(None)]
    else:
        # tind.stop - tind.start is the total number of t-indices ignoring the step.
        # Adding tind.step - 1 and integer-dividing by tind.step converts to the total
        # number accounting for the step.
        nt = (tind.stop - tind.start + tind.step - 1) // tind.step
        n_outputs = (nt + time_split_size - 1) // time_split_size
        filenames = []
        t_slices = []
        for i in range(n_outputs):
            parts = fullpath.split(".")
            parts[-2] += str(time_split_first_label + i)
            filename = ".".join(parts)
            filenames.append(filename)
            t_slices.append(slice(i * time_split_size, (i + 1) * time_split_size))
        return filenames, t_slices


if __name__ == "__main__":
    from .scripts.bout_squashoutput import main

    main()
