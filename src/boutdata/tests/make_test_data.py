from copy import copy

import numpy as np
from netCDF4 import Dataset

field3d_t_list = ["field3d_t_1"]
field3d_list = ["field3d_1"]
field2d_t_list = ["field2d_t_1"]
field2d_list = ["field2d_1"]
fieldperp_t_list = ["fieldperp_t_1"]
fieldperp_list = ["fieldperp_1"]
scalar_t_list = ["t_array", "scalar_t_1"]

# Note "yindex_global" attribute not included here for FieldPerps, because it is handled
# specially
expected_attributes = {
    "field3d_t_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "field3d_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "field2d_t_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Average",
    },
    "field2d_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Average",
    },
    "fieldperp_t_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
    "fieldperp_1": {
        "cell_location": "CELL_CENTRE",
        "direction_y": "Standard",
        "direction_z": "Standard",
    },
}

expected_file_attributes = {
    "global_str_attribute": "foobar",
    "global_int_attribute": 42,
    "global_float_attribute": 7.0,
}


def make_grid_info(
    *, mxg=2, myg=2, nxpe=1, nype=1, ixseps1=None, ixseps2=None, xpoints=0
):
    """
    Create a dict of parameters used for creating test data

    Parameters
    ----------
    mxg : int, optional
        Number of guard cells in the x-direction
    myg : int, optional
        Number of guard cells in the y-direction
    nxpe : int, optional
        Number of processes in the x-direction
    nype : int, optional
        Number of processes in the y-direction
    ixseps1 : int, optional
        x-index (where indexing includes boundary points) of point just outside
        first separatrix
    ixseps2 : int, optional
        x-index (where indexing includes boundary points) of point just outside
        second separatrix
    xpoints : int, optional
        Number of X-points.
    """
    grid_info = {}
    grid_info["iteration"] = 6
    grid_info["MXSUB"] = 3
    grid_info["MYSUB"] = 4
    grid_info["MZSUB"] = 5
    grid_info["MXG"] = mxg
    grid_info["MYG"] = myg
    grid_info["MZG"] = 0
    grid_info["NXPE"] = nxpe
    grid_info["NYPE"] = nype
    grid_info["NZPE"] = 1
    grid_info["nx"] = nxpe * grid_info["MXSUB"] + 2 * mxg
    grid_info["ny"] = nype * grid_info["MYSUB"]
    grid_info["nz"] = grid_info["NZPE"] * grid_info["MZSUB"]
    grid_info["MZ"] = grid_info["nz"]
    if ixseps1 is None:
        grid_info["ixseps1"] = grid_info["nx"]
    else:
        grid_info["ixseps1"] = ixseps1
    if ixseps2 is None:
        grid_info["ixseps2"] = grid_info["nx"]
    else:
        grid_info["ixseps2"] = ixseps2
    if xpoints == 0:
        grid_info["jyseps1_1"] = -1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = grid_info["ny"]
    elif xpoints == 1:
        if nype < 3:
            raise ValueError(f"nype={nype} not enough for single-null")
        yproc_per_region = nype // 3
        grid_info["jyseps1_1"] = yproc_per_region * grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = grid_info["ny"] // 2 - 1
        grid_info["ny_inner"] = grid_info["ny"] // 2
        grid_info["jyseps1_2"] = grid_info["ny"] // 2 - 1
        grid_info["jyseps2_2"] = 2 * yproc_per_region * grid_info["MYSUB"] - 1
    elif xpoints == 2:
        if nype < 6:
            raise ValueError(f"nype={nype} not enough for single-null")
        yproc_per_region = nype // 6
        grid_info["jyseps1_1"] = yproc_per_region * grid_info["MYSUB"] - 1
        grid_info["jyseps2_1"] = 2 * yproc_per_region * grid_info["MYSUB"] - 1
        grid_info["ny_inner"] = 3 * yproc_per_region * grid_info["MYSUB"]
        grid_info["jyseps1_2"] = 4 * yproc_per_region * grid_info["MYSUB"] - 1
        grid_info["jyseps2_2"] = 5 * yproc_per_region * grid_info["MYSUB"] - 1
    else:
        raise ValueError(f"Unsupported value for xpoints: {xpoints}")

    return grid_info


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
            raise ValueError(f"Unexpected boundary input {b}")
    xinner = "xinner" in boundaries
    xouter = "xouter" in boundaries
    ylower = "ylower" in boundaries
    yupper = "yupper" in boundaries

    with Dataset(tmpdir.joinpath(f"BOUT.dmp.{i}.nc"), "w") as outputfile:
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

        def create3D(name):
            var = outputfile.createVariable(name, float, ("x", "y", "z"))

            data = rng.random((localnx, localny, localnz))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)

            result[name] = data[xslice, yslice, zslice]

        create3D("field3d_1")

        # Field2D
        def create2D_t(name):
            var = outputfile.createVariable(name, float, ("t", "x", "y"))

            data = rng.random((nt, localnx, localny))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)

            result[name] = data[:, xslice, yslice]

        create2D_t("field2d_t_1")

        def create2D(name):
            var = outputfile.createVariable(name, float, ("x", "y"))

            data = rng.random((localnx, localny))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)

            result[name] = data[xslice, yslice]

        create2D("field2d_1")

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

        def createPerp(name):
            var = outputfile.createVariable(name, float, ("x", "z"))

            data = rng.random((localnx, localnz))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)
            var.setncattr("yindex_global", fieldperp_global_yind)

            result[name] = data[xslice, zslice]

        createPerp("fieldperp_1")

        # Time-dependent array
        def createScalar_t(name):
            var = outputfile.createVariable(name, float, ("t",))

            data = rng.random(nt)
            var[:] = data

            result[name] = data

        createScalar_t("t_array")
        createScalar_t("scalar_t_1")

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

        for attrname, attr in expected_file_attributes.items():
            setattr(outputfile, attrname, attr)

    return result


def create_restart_file(*, i, tmpdir, rng, grid_info, fieldperp_global_yind):
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
    fieldperp_global_yind : int
        Global y-index for a FieldPerp (should be -1 if FieldPerp is not on this
        processor).

    Returns
    -------
    Dict of scalars and numpy arrays
    """
    mxg = grid_info["MXG"]
    myg = grid_info["MYG"]
    mzg = grid_info["MZG"]
    localnx = grid_info["MXSUB"] + 2 * mxg
    localny = grid_info["MYSUB"] + 2 * myg
    localnz = grid_info["MZSUB"] + 2 * mzg

    with Dataset(tmpdir.joinpath(f"BOUT.restart.{i}.nc"), "w") as outputfile:
        outputfile.createDimension("x", localnx)
        outputfile.createDimension("y", localny)
        outputfile.createDimension("z", localnz)

        # Create slices for returned data without guard cells
        xslice = slice(mxg, None if mxg == 0 else -mxg)
        yslice = slice(myg, None if myg == 0 else -myg)
        zslice = slice(mzg, None if mzg == 0 else -mzg)

        result = {}

        # Field3D
        def create3D(name):
            var = outputfile.createVariable(name, float, ("x", "y", "z"))

            data = rng.random((localnx, localny, localnz))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)

            result[name] = data[xslice, yslice, zslice]

        create3D("field3d_1")

        # Field2D
        def create2D(name):
            var = outputfile.createVariable(name, float, ("x", "y"))

            data = rng.random((localnx, localny))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)

            result[name] = data[xslice, yslice]

        create2D("field2d_1")

        # FieldPerp
        def createPerp(name):
            var = outputfile.createVariable(name, float, ("x", "z"))

            data = rng.random((localnx, localnz))
            var[:] = data
            for key, value in expected_attributes[name].items():
                var.setncattr(key, value)
            var.setncattr("yindex_global", fieldperp_global_yind)

            result[name] = data[xslice, zslice]

        createPerp("fieldperp_1")

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


def concatenate_data(data_list, *, nxpe, fieldperp_yproc_ind, has_t_dim=True):
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
        raise ValueError(f"nxpe={nxpe} does not divide len(data_list)={npes}")

    if has_t_dim:
        for var in ("field3d_t_1", "field2d_t_1"):
            # Join in x-direction
            parts = [
                np.concatenate(
                    [data_list[j][var] for j in range(i * nxpe, (i + 1) * nxpe)], axis=1
                )
                for i in range(nype)
            ]
            # Join in y-direction
            result[var] = np.concatenate(parts, axis=2)

    for var in ("field3d_1", "field2d_1"):
        # Join in x-direction
        parts = [
            np.concatenate(
                [data_list[j][var] for j in range(i * nxpe, (i + 1) * nxpe)], axis=0
            )
            for i in range(nype)
        ]
        # Join in y-direction
        result[var] = np.concatenate(parts, axis=1)

    if has_t_dim:
        for var in ("fieldperp_t_1",):
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

    for var in ("fieldperp_1",):
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
