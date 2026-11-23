import pytest

import boututils.datafile


@pytest.fixture(params=["h5py", "netcdf4"])
def netcdf_reader(request, monkeypatch):
    """Run a test with each of the libraries that can be used to read NetCDF files"""
    if request.param == "h5py":
        pytest.importorskip("h5py")
    monkeypatch.setattr(boututils.datafile, "netcdf_reader", request.param)
    return request.param
