import subprocess
import sys

import numpy as np
import pytest
from netCDF4 import Dataset, default_fillvals

import boututils.datafile
from boutdata.tests.make_test_data import (
    create_dump_file,
    create_restart_file,
    make_grid_info,
)
from boututils.datafile import DataFile, DataFile_netCDF

h5py = pytest.importorskip("h5py")
from boututils.datafile import DataFile_netCDF_h5py


def create_unusual_file(path):
    """Create a NetCDF-4 file with features not found in the BOUT++ test files"""
    with Dataset(path, "w") as f:
        f.createDimension("t", None)
        f.createDimension("x", 4)
        f.createDimension("y", 3)
        f.createDimension("char5", 5)
        f.setncattr("title", "unusual file")
        f.setncattr("numbers", np.array([1.0, 2.0]))
        f.setncattr("one_number", np.int32(3))
        f.setncattr_string("strings", ["a", "bc"])

        # Time-dependent variable with fewer time points than another variable
        v = f.createVariable("short_t", float, ("t", "x"))
        v[:] = np.arange(8.0).reshape(2, 4)
        v = f.createVariable("long_t", float, ("t", "y"))
        v[:] = np.arange(9.0).reshape(3, 3)
        v.setncattr("multiple", np.array([1, 2, 3], dtype=np.int32))
        v.setncattr("empty_string", "")
        v.setncattr_string("vlen_string", "vlen")

        # Coordinate variable, which is its own dimension
        v = f.createVariable("t", float, ("t",))
        v[:] = [0.0, 1.0, 2.0]

        # Variable with the name of a dimension, but different dimensions
        v = f.createVariable("y", float, ("x",))
        v[:] = np.arange(4.0)

        # Fill values, with and without the _FillValue attribute
        v = f.createVariable("with_fill", float, ("x", "y"), fill_value=-1.0)
        v[:] = np.arange(12.0).reshape(4, 3)
        v[1, 1] = -1.0
        v = f.createVariable("default_fill", float, ("x", "y"))
        data = np.arange(12.0).reshape(4, 3)
        data[2, 0] = default_fillvals["f8"]
        v[:] = data
        v = f.createVariable("with_nan_fill", float, ("x",), fill_value=np.nan)
        v[:] = [0.0, np.nan, 2.0, 3.0]
        v = f.createVariable("byte_var", np.int8, ("x",))
        v[:] = [1, 2, 3, 4]

        # Variables that netCDF4 converts when reading
        v = f.createVariable("packed", np.int16, ("x",))
        v.setncattr("scale_factor", 0.5)
        v.setncattr("add_offset", 1.0)
        v[:] = [1.0, 2.0, 3.0, 4.0]

        # Strings
        v = f.createVariable("string_scalar", str, ())
        v[0] = "hello"
        v = f.createVariable("string_array", str, ("x",))
        v[:] = np.array(["a", "bb", "ccc", "dddd"], dtype=object)
        v = f.createVariable("chars", "S1", ("t", "char5"))
        v[:] = np.array([list("abcde"), list("fghij"), list("klmno")], dtype="S1")

        # Variables in a group are not listed by netCDF4
        g = f.createGroup("section")
        v = g.createVariable("in_group", float, ("x",))
        v[:] = np.arange(4.0)

        v = f.createVariable("int_scalar", np.int32, ())
        v.assignValue(7)


@pytest.fixture(scope="module")
def test_files(tmp_path_factory):
    """NetCDF-4 files to compare the two readers on"""
    tmp_path = tmp_path_factory.mktemp("datafile")
    grid_info = make_grid_info()
    rng = np.random.default_rng(42)
    create_dump_file(
        i=0,
        tmpdir=tmp_path,
        rng=rng,
        grid_info=grid_info,
        boundaries=["xinner", "xouter", "ylower", "yupper"],
        fieldperp_global_yind=3,
    )
    create_restart_file(
        i=0, tmpdir=tmp_path, rng=rng, grid_info=grid_info, fieldperp_global_yind=3
    )
    create_unusual_file(tmp_path / "unusual.nc")
    return {
        "dump": tmp_path / "BOUT.dmp.0.nc",
        "restart": tmp_path / "BOUT.restart.0.nc",
        "unusual": tmp_path / "unusual.nc",
    }


def assert_same(expected, actual, what):
    """Check that two results are identical, including their types"""
    assert type(actual) is type(expected), what
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, what
        assert actual.shape == expected.shape, what
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected), err_msg=what
        )
        np.testing.assert_array_equal(
            np.ma.getmaskarray(actual), np.ma.getmaskarray(expected), err_msg=what
        )
        if hasattr(expected, "attributes"):
            assert_same(expected.attributes, actual.attributes, f"{what}.attributes")
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), what
        for key in expected:
            assert_same(expected[key], actual[key], f"{what}[{key}]")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), what
        for i, (e, a) in enumerate(zip(expected, actual)):
            assert_same(e, a, f"{what}[{i}]")
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual), what
    else:
        assert actual == expected, what


def call_or_error(method, *args, **kwargs):
    """Result of a call, or the name of the exception it raised"""
    try:
        return method(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


@pytest.mark.parametrize("file", ["dump", "restart", "unusual"])
def test_h5py_reader_matches_netcdf4(test_files, file):
    path = test_files[file]
    with DataFile_netCDF(path) as expected, DataFile_netCDF_h5py(path) as actual:
        check_same_contents(expected, actual)


def check_same_contents(expected, actual):
    assert_same(expected.list(), actual.list(), "list()")
    assert_same(
        expected.list_file_attributes(),
        actual.list_file_attributes(),
        "list_file_attributes()",
    )
    for name in expected.list_file_attributes():
        assert_same(
            expected.read_file_attribute(name),
            actual.read_file_attribute(name),
            f"read_file_attribute({name})",
        )
    with pytest.raises(AttributeError):
        actual.read_file_attribute("not_an_attribute")

    for name in expected.list():
        assert_same(expected.dimensions(name), actual.dimensions(name), f"{name} dims")
        assert_same(expected.size(name), actual.size(name), f"{name} size")
        assert_same(expected.ndims(name), actual.ndims(name), f"{name} ndims")
        assert_same(expected.attributes(name), actual.attributes(name), f"{name} attrs")
        for asBoutArray in [True, False]:
            assert_same(
                expected.read(name, asBoutArray=asBoutArray),
                actual.read(name, asBoutArray=asBoutArray),
                f"{name} read(asBoutArray={asBoutArray})",
            )

        ndims = expected.ndims(name)
        if ndims == 0:
            continue
        ranges_list = [
            [slice(1, None, 2)] + [slice(0, 2)] * (ndims - 1),
            [slice(0, 1)] + [None] * (ndims - 1),
            [0, 1] * ndims,
            [slice(None, None, -1)] * ndims,
        ]
        for ranges in ranges_list:
            for asBoutArray in [True, False]:
                assert_same(
                    call_or_error(expected.read, name, ranges, asBoutArray),
                    call_or_error(actual.read, name, ranges, asBoutArray),
                    f"{name} read({ranges}, asBoutArray={asBoutArray})",
                )

    # Variables that do not exist, or are only NetCDF dimensions
    for name in ["not_a_variable", "x", "section", "section/in_group"]:
        assert actual.read(name) is None, name
        assert actual.size(name) == [], name
        with pytest.raises(ValueError):
            actual.dimensions(name)


def test_h5py_reader_case_insensitive(test_files):
    path = test_files["dump"]
    with DataFile_netCDF(path) as expected, DataFile_netCDF_h5py(path) as actual:
        assert_same(expected.read("FIELD3D_T_1"), actual.read("FIELD3D_T_1"), "read")
        assert_same(
            call_or_error(expected.attributes, "FIELD3D_T_1"),
            call_or_error(actual.attributes, "FIELD3D_T_1"),
            "attributes",
        )


def test_h5py_reader_is_read_only(test_files):
    with DataFile_netCDF_h5py(test_files["dump"]) as f:
        with pytest.raises(Exception, match="File not writeable"):
            f.write("new_variable", np.zeros(3))
        with pytest.raises(Exception, match="File not writeable"):
            f.write_file_attribute("new_attribute", 1)
    with pytest.raises(ValueError):
        DataFile_netCDF_h5py(test_files["dump"], write=True)


def test_h5py_reader_attributes_are_copies(test_files):
    """Changing returned attributes must not change the cached metadata"""
    with DataFile_netCDF_h5py(test_files["dump"]) as f:
        first = f.attributes("field3d_t_1")
    first["new_attribute"] = 1
    with DataFile_netCDF_h5py(test_files["dump"]) as f:
        second = f.attributes("field3d_t_1")
    assert "new_attribute" not in second


def test_h5py_reader_sees_changed_file(tmp_path):
    """Cached metadata must not be used after a file has been changed"""
    path = tmp_path / "changing.nc"
    with Dataset(path, "w") as f:
        f.createDimension("x", 2)
        f.createVariable("a", float, ("x",))[:] = [1.0, 2.0]
    with DataFile_netCDF_h5py(path) as f:
        assert f.list() == ["a"]

    with Dataset(path, "a") as f:
        f.createVariable("b", float, ("x",))[:] = [3.0, 4.0]
    with DataFile_netCDF_h5py(path) as f:
        assert f.list() == ["a", "b"]


@pytest.fixture
def netcdf3_file(tmp_path):
    path = tmp_path / "netcdf3.nc"
    with Dataset(path, "w", format="NETCDF3_CLASSIC") as f:
        f.createDimension("x", 2)
        f.createVariable("a", float, ("x",))[:] = [1.0, 2.0]
    return path


def test_dispatch_auto(test_files, netcdf3_file, monkeypatch):
    monkeypatch.setattr(boututils.datafile, "netcdf_reader", "auto")
    with DataFile(test_files["dump"]) as f:
        assert type(f.impl) is DataFile_netCDF_h5py
    # NetCDF-3 files are not HDF5 files
    with DataFile(netcdf3_file) as f:
        assert type(f.impl) is DataFile_netCDF
    # Writing needs netCDF4
    with DataFile(netcdf3_file, write=True) as f:
        assert type(f.impl) is DataFile_netCDF
    with DataFile(test_files["dump"].parent / "new.nc", create=True) as f:
        assert type(f.impl) is DataFile_netCDF

    monkeypatch.setattr(boututils.datafile, "has_h5py", False)
    with DataFile(test_files["dump"]) as f:
        assert type(f.impl) is DataFile_netCDF


def test_dispatch_netcdf4(test_files, monkeypatch):
    monkeypatch.setattr(boututils.datafile, "netcdf_reader", "netcdf4")
    with DataFile(test_files["dump"]) as f:
        assert type(f.impl) is DataFile_netCDF


def test_dispatch_h5py(test_files, monkeypatch):
    monkeypatch.setattr(boututils.datafile, "netcdf_reader", "h5py")
    with DataFile(test_files["dump"]) as f:
        assert type(f.impl) is DataFile_netCDF_h5py

    monkeypatch.setattr(boututils.datafile, "has_h5py", False)
    with pytest.raises(ImportError):
        DataFile(test_files["dump"])


def test_dispatch_invalid(test_files, monkeypatch):
    monkeypatch.setattr(boututils.datafile, "netcdf_reader", "not_a_library")
    with pytest.raises(ValueError):
        DataFile(test_files["dump"])


@pytest.mark.parametrize(
    ("reader", "expected"),
    [("netcdf4", "DataFile_netCDF"), ("auto", "DataFile_netCDF_h5py")],
)
def test_dispatch_environment_variable(test_files, monkeypatch, reader, expected):
    monkeypatch.setenv("BOUTDATA_NETCDF_READER", reader)
    code = (
        "from boututils.datafile import DataFile; "
        f"print(type(DataFile({str(test_files['dump'])!r}).impl).__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected
