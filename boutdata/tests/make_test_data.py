from copy import copy
from netCDF4 import Dataset
import numpy as np

field3d_t_list = ["field3d_t_1", "field3d_t_2"]
field3d_list = ["field3d_1", "field3d_2"]
field2d_t_list = ["field2d_t_1", "field2d_t_2"]
field2d_list = ["field2d_1", "field2d_2"]
fieldperp_t_list = ["fieldperp_t_1", "fieldperp_t_2"]
fieldperp_list = ["fieldperp_1", "fieldperp_2"]
scalar_t_list = ["t_array", "scalar_t_1", "scalar_t_2"]

# Note "yindex_global" attribute not included here for FieldPerps, because it is handled
# specially
expected_attributes = {
    "field3d_t_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "field3d_t_2": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "field3d_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "field3d_2": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "field2d_t_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Average",
    },
    "field2d_t_2": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Average",
    },
    "field2d_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Average",
    },
    "field2d_2": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Average",
    },
    "fieldperp_t_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "fieldperp_t_2": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "fieldperp_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "fieldperp_2": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
}


def create_dump_file(*, i, tmpdir, rng, grid_info, boundaries, fieldperp_global_yind):
    """
    Create a netCDF file mocking up a BOUT++ output file, and also return the data
    without guard cells

    Parameters
    ----------
    i : int
        Number of the output file
    tmpdir : pathlib.Path
        Directory to write the dump file in
    rng : numpy.random.Generator
        Random number generator to create data
    grid_info : dict
        Dictionary containing grid sizes, etc
    boundaries : sequence of str
        Which edges are boundaries. Should be a sequence containing any of "xinner",
        "xouter", "ylower" and "yupper".
    fieldperp_global_yind : int
        Global y-index for a FieldPerp (should be -1 if FieldPerp is not on this
        processor).

    Returns
    -------
    Dict of scalars and numpy arrays
    """
    nt = grid_info["iteration"]
    mxg = grid_info["MXG"]
    myg = grid_info["MYG"]
    mzg = grid_info["MZG"]
    localnx = grid_info["MXSUB"] + 2 * mxg
    localny = grid_info["MYSUB"] + 2 * myg
    localnz = grid_info["MZSUB"] + 2 * mzg

    for b in boundaries:
        if b not in ("xinner", "xouter", "yupper", "ylower"):
            raise ValueError("Unexpected boundary input " + str(b))
    xinner = "xinner" in boundaries
    xouter = "xouter" in boundaries
    ylower = "ylower" in boundaries
    yupper = "yupper" in boundaries

    outputfile = Dataset(tmpdir.joinpath("BOUT.dmp." + str(i) + ".nc"), "w")

    outputfile.createDimension("t", None)
    outputfile.createDimension("x", localnx)
    outputfile.createDimension("y", localny)
    outputfile.createDimension("z", localnz)

    # Create slices for returned data without guard cells
    xslice = slice(None if xinner else mxg, None if xouter or mxg == 0 else -mxg)
    yslice = slice(None if ylower else myg, None if yupper or myg == 0 else -myg)
    zslice = slice(mzg, None if mzg == 0 else -mzg)

    result = {}

    # Field3D
    def create3D_t(name):
        var = outputfile.createVariable(name, float, ("t", "x", "y", "z"))

        data = rng.random((nt, localnx, localny, localnz))
        var[:] = data
        for key, value in expected_attributes[name].items():
            var.setncattr(key, value)

        result[name] = data[:, xslice, yslice, zslice]

    create3D_t("field3d_t_1")
    create3D_t("field3d_t_2")

    def create3D(name):
        var = outputfile.createVariable(name, float, ("x", "y", "z"))

        data = rng.random((localnx, localny, localnz))
        var[:] = data
        for key, value in expected_attributes[name].items():
            var.setncattr(key, value)

        result[name] = data[xslice, yslice, zslice]

    create3D("field3d_1")
    create3D("field3d_2")

    # Field2D
    def create2D_t(name):
        var = outputfile.createVariable(name, float, ("t", "x", "y"))

        data = rng.random((nt, localnx, localny))
        var[:] = data
        for key, value in expected_attributes[name].items():
            var.setncattr(key, value)

        result[name] = data[:, xslice, yslice]

    create2D_t("field2d_t_1")
    create2D_t("field2d_t_2")

    def create2D(name):
        var = outputfile.createVariable(name, float, ("x", "y"))

        data = rng.random((localnx, localny))
        var[:] = data
        for key, value in expected_attributes[name].items():
            var.setncattr(key, value)

        result[name] = data[xslice, yslice]

    create2D("field2d_1")
    create2D("field2d_2")

    # FieldPerp
    def createPerp_t(name):
        var = outputfile.createVariable(name, float, ("t", "x", "z"))

        data = rng.random((nt, localnx, localnz))
        var[:] = data
        for key, value in expected_attributes[name].items():
            var.setncattr(key, value)
        var.setncattr("yindex_global", fieldperp_global_yind)

        result[name] = data[:, xslice, zslice]

    createPerp_t("fieldperp_t_1")
    createPerp_t("fieldperp_t_2")

    def createPerp(name):
        var = outputfile.createVariable(name, float, ("x", "z"))

        data = rng.random((localnx, localnz))
        var[:] = data
        for key, value in expected_attributes[name].items():
            var.setncattr(key, value)
        var.setncattr("yindex_global", fieldperp_global_yind)

        result[name] = data[xslice, zslice]

    createPerp("fieldperp_1")
    createPerp("fieldperp_2")

    # Time-dependent array
    def createScalar_t(name):
        var = outputfile.createVariable(name, float, ("t",))

        data = rng.random(nt)
        var[:] = data

        result[name] = data

    createScalar_t("t_array")
    createScalar_t("scalar_t_1")
    createScalar_t("scalar_t_2")

    # Scalar
    def createScalar(name, value):
        var = outputfile.createVariable(name, type(value))

        var[...] = value

        result[name] = value

    createScalar("BOUT_VERSION", 4.31)
    for key, value in grid_info.items():
        createScalar(key, value)
    nxpe = grid_info["NXPE"]
    createScalar("PE_XIND", i % nxpe)
    createScalar("PE_YIND", i // nxpe)
    createScalar("MYPE", i)

    return result


def concatenate_data(data_list, *, nxpe, fieldperp_yproc_ind):
    """
    Joins together lists of data arrays for expected results from each process into a
    global array.

    Parameters
    ----------
    data_list : list of dict of {str: numpy array}
        List, ordered by processor number, of dicts of expected data (key is name, value
        is scalar or numpy array of data). Data should not include guard cells.
    nxpe : int
        Number of processes in the x-direction.
    fieldperp_yproc_ind : int
        y-processes index to keep FieldPerps from. FieldPerps can only be defined at a
        single global y-index, so should be discarded from other processes.
    """
    # Just keep scalars from root file
    result = copy(data_list[0])
    for x in list(result.keys()):
        if x[:5] == "field":
            result.pop(x)

    npes = len(data_list)
    nype = npes // nxpe
    if npes % nxpe != 0:
        raise ValueError("nxpe=%i does not divide len(data_list)=%i".format(nxpe, npes))

    for var in ("field3d_t_1", "field3d_t_2", "field2d_t_1", "field2d_t_2"):
        # Join in x-direction
        parts = [
            np.concatenate(
                [data_list[j][var] for j in range(i * nxpe, (i + 1) * nxpe)], axis=1
            )
            for i in range(nype)
        ]
        # Join in y-direction
        result[var] = np.concatenate(parts, axis=2)

    for var in ("field3d_1", "field3d_2", "field2d_1", "field2d_2"):
        # Join in x-direction
        parts = [
            np.concatenate(
                [data_list[j][var] for j in range(i * nxpe, (i + 1) * nxpe)], axis=0
            )
            for i in range(nype)
        ]
        # Join in y-direction
        result[var] = np.concatenate(parts, axis=1)

    for var in ("fieldperp_t_1", "fieldperp_t_2"):
        # Join in x-direction
        result[var] = np.concatenate(
            [
                data_list[j][var]
                for j in range(
                    fieldperp_yproc_ind * nxpe, (fieldperp_yproc_ind + 1) * nxpe
                )
            ],
            axis=1,
        )

    for var in ("fieldperp_1", "fieldperp_2"):
        # Join in x-direction
        result[var] = np.concatenate(
            [
                data_list[j][var]
                for j in range(
                    fieldperp_yproc_ind * nxpe, (fieldperp_yproc_ind + 1) * nxpe
                )
            ],
            axis=0,
        )

    return result


def apply_slices(expected, tslice, xslice, yslice, zslice):
    """
    Slice expected data

    Parameters
    ----------
    expected : dict {str: numpy array}
        dict of expected data (key is name, value is scalar or numpy array of data).
        Arrays should be global (not per-process).
    tslice : slice object
        Slice to apply in the t-direction
    xslice : slice object
        Slice to apply in the x-direction
    yslice : slice object
        Slice to apply in the y-direction
    zslice : slice object
        Slice to apply in the z-direction
    """
    # Note - this should by called after 'xguards' and 'yguards' have been applied to
    # 'expected'.
    for varname in field3d_t_list:
        expected[varname] = expected[varname][tslice, xslice, yslice, zslice]
    for varname in field3d_list:
        expected[varname] = expected[varname][xslice, yslice, zslice]
    for varname in field2d_t_list:
        expected[varname] = expected[varname][tslice, xslice, yslice]
    for varname in field2d_list:
        expected[varname] = expected[varname][xslice, yslice]
    for varname in fieldperp_t_list:
        expected[varname] = expected[varname][tslice, xslice, zslice]
    for varname in fieldperp_list:
        expected[varname] = expected[varname][xslice, zslice]
    for varname in scalar_t_list:
        expected[varname] = expected[varname][tslice]


def remove_xboundaries(expected, mxg):
    """
    Remove x-boundary points from expected data

    Parameters
    ----------
    expected : dict {str: numpy array}
        dict of expected data (key is name, value is scalar or numpy array of data).
        Arrays should be global (not per-process).
    mxg : int
        Number of boundary points to remove.
    """
    if mxg == 0:
        return

    for varname in field3d_t_list + field2d_t_list + fieldperp_t_list:
        expected[varname] = expected[varname][:, mxg:-mxg]

    for varname in field3d_list + field2d_list + fieldperp_list:
        expected[varname] = expected[varname][mxg:-mxg]


def remove_yboundaries(expected, myg, ny_inner, doublenull):
    """
    Remove y-boundary points from expected data

    Parameters
    ----------
    expected : dict {str: numpy array}
        dict of expected data (key is name, value is scalar or numpy array of data).
        Arrays should be global (not per-process).
    myg : int
        Number of boundary points to remove.
    ny_inner : int
        BOUT++ ny_inner parameter - specifies location of 'upper target' y-boundary for
        double-null topology
    doublenull : bool
        If True the data for double-null. If False the data is for single-null, limiter,
        core or SOL topologies which do not have a y-boundary in the middle of the
        domain.
    """
    if myg == 0:
        return

    if doublenull:
        for varname in field3d_t_list + field2d_t_list:
            expected[varname] = np.concatenate(
                [
                    expected[varname][:, :, myg : ny_inner + myg],
                    expected[varname][:, :, ny_inner + 3 * myg : -myg],
                ],
                axis=2,
            )
        for varname in field3d_list + field2d_list:
            expected[varname] = np.concatenate(
                [
                    expected[varname][:, myg : ny_inner + myg],
                    expected[varname][:, ny_inner + 3 * myg : -myg],
                ],
                axis=1,
            )
    else:
        for varname in field3d_t_list + field2d_t_list:
            expected[varname] = expected[varname][:, :, myg:-myg]
        for varname in field3d_list + field2d_list:
            expected[varname] = expected[varname][:, myg:-myg]


def remove_yboundaries_upper_divertor(expected, myg, ny_inner):
    """
    Remove y-boundary points just from the 'upper divertor' - the y-boundaries in the
    middle of the domain.

    Parameters
    ----------
    expected : dict {str: numpy array}
        dict of expected data (key is name, value is scalar or numpy array of data).
        Arrays should be global (not per-process).
    myg : int
        Number of boundary points to remove.
    ny_inner : int
        BOUT++ ny_inner parameter - specifies location of 'upper target' y-boundary for
        double-null topology
    """
    if myg == 0:
        return

    for varname in field3d_t_list + field2d_t_list:
        expected[varname] = np.concatenate(
            [
                expected[varname][:, :, : ny_inner + myg],
                expected[varname][:, :, ny_inner + 3 * myg :],
            ],
            axis=2,
        )

    for varname in field3d_list + field2d_list:
        expected[varname] = np.concatenate(
            [
                expected[varname][:, : ny_inner + myg],
                expected[varname][:, ny_inner + 3 * myg :],
            ],
            axis=1,
        )