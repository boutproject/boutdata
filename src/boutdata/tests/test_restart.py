from glob import glob

import numpy as np
import numpy.testing as npt
import pytest

from boutdata import restart
from boutdata.collect import collect
from boutdata.tests.make_test_data import (
    concatenate_data,
    create_restart_file,
    expected_attributes,
    make_grid_info,
)

# Run every test with each library that can read NetCDF files
pytestmark = pytest.mark.usefixtures("netcdf_reader")


def check_redistributed_data(
    expected,
    *,
    fieldperp_global_yind,
    path,
):
    """
    Use `collect()` to read 'actual' data from the files. Test that 'actual' and
    'expected' data and attributes match.

    Parameters
    ----------
    expected : dict {str: numpy array}
        dict of expected data (key is name, value is scalar or numpy array of data).
        Arrays should be global (not per-process).
    fieldperp_global_yind : int
        Global y-index where FieldPerps are expected to be defined.
    path : pathlib.Path or str
        Path to collect data from.
    """
    expected_different = ["MXSUB", "MYSUB", "NXPE", "NYPE"]
    for varname in expected:
        if varname in expected_different:
            # These variables are expected to be changed by restart.redistribute
            continue
        actual = collect(
            varname, path=path, prefix="BOUT.restart", xguards=False, yguards=False
        )
        npt.assert_array_equal(expected[varname], actual)
        actual_keys = list(actual.attributes.keys())
        if varname in expected_attributes:
            for a in expected_attributes[varname]:
                assert actual.attributes[a] == expected_attributes[varname][a]
                actual_keys.remove(a)

        if "fieldperp" in varname:
            assert actual.attributes["yindex_global"] == fieldperp_global_yind
            actual_keys.remove("yindex_global")

        assert actual_keys == ["bout_type"]

        if "field3d" in varname:
            assert actual.attributes["bout_type"] == "Field3D"
        elif "field2d" in varname:
            assert actual.attributes["bout_type"] == "Field2D"
        elif "fieldperp" in varname:
            assert actual.attributes["bout_type"] == "FieldPerp"
        else:
            assert actual.attributes["bout_type"] == "scalar"


class TestRestart:
    def test_redistribute_connected_doublenull(self, tmp_path):
        """
        Check for a connected double-null case using a large number of processes.
        'Large' means there is at least one process in each region with no edges
        touching another region.
        """
        npes_redistributed = 6

        grid_info = make_grid_info(nxpe=3, nype=18, ixseps1=7, ixseps2=7, xpoints=2)

        fieldperp_global_yind = 19
        fieldperp_yproc_ind = 4

        rng = np.random.default_rng(108)

        restart_params = [
            # inner, lower divertor leg
            (0, -1),
            (1, -1),
            (2, -1),
            (3, -1),
            (4, -1),
            (5, -1),
            (6, -1),
            (7, -1),
            (8, -1),
            # inner core
            (9, -1),
            (10, -1),
            (11, -1),
            (12, fieldperp_global_yind),
            (13, fieldperp_global_yind),
            (14, fieldperp_global_yind),
            (15, -1),
            (16, -1),
            (17, -1),
            # inner, upper divertor leg
            (18, -1),
            (19, -1),
            (20, -1),
            (21, -1),
            (22, -1),
            (23, -1),
            (24, -1),
            (25, -1),
            (26, -1),
            # outer, upper divertor leg
            (27, -1),
            (28, -1),
            (29, -1),
            (30, -1),
            (31, -1),
            (32, -1),
            (33, -1),
            (34, -1),
            (35, -1),
            # outer core
            (36, -1),
            (37, -1),
            (38, -1),
            (39, -1),
            (40, -1),
            (41, -1),
            (42, -1),
            (43, -1),
            (44, -1),
            # outer, lower divertor leg
            (45, -1),
            (46, -1),
            (47, -1),
            (48, -1),
            (49, -1),
            (50, -1),
            (51, -1),
            (52, -1),
            (53, -1),
        ]
        restarts = []
        for i, fieldperp_yind in restart_params:
            restarts.append(
                create_restart_file(
                    tmpdir=tmp_path,
                    rng=rng,
                    grid_info=grid_info,
                    i=i,
                    fieldperp_global_yind=fieldperp_yind,
                )
            )

        expected = concatenate_data(
            restarts,
            nxpe=grid_info["NXPE"],
            fieldperp_yproc_ind=fieldperp_yproc_ind,
            has_t_dim=False,
        )

        new_path = tmp_path.joinpath("new_restarts")
        new_path.mkdir()
        restart.redistribute(npes=npes_redistributed, path=tmp_path, output=new_path)

        # Check the right number of files have been created
        new_files = glob(str(new_path.joinpath("*")))
        assert len(new_files) == npes_redistributed

        # Check data in redistributed restart files
        check_redistributed_data(
            expected,
            fieldperp_global_yind=fieldperp_global_yind,
            path=new_path,
        )
