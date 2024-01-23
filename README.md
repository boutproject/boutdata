# boutdata

[![Build Status](https://travis-ci.org/boutproject/boutdata.svg?branch=master)](https://travis-ci.org/boutproject/boutdata)
[![codecov](https://codecov.io/gh/boutproject/boutdata/branch/master/graph/badge.svg)](https://codecov.io/gh/boutproject/boutdata)
[![Python](https://img.shields.io/badge/python->=3.6-blue.svg)](https://www.python.org/)
[![pypi package](https://badge.fury.io/py/boutdata.svg)](https://pypi.org/project/boutdata/)
[![PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/boutproject/boutdata/blob/master/LICENSE)

Python tools for working with [BOUT++](https://github.com/boutproject/BOUT-dev.git).

> **NOTE**: This package will likely be superseded by
  [`xBOUT`](https://github.com/boutproject/xBOUT) in the near future

# Dependencies

`boutdata` uses [`netcfd4`](https://github.com/Unidata/netcdf4-python)
which requires [`HDF5`](https://www.h5py.org) and
[`netcdf-4`](https://github.com/Unidata/netcdf-c/releases) are
installed, and that the `nc-config` utility is in your `PATH`. This
can be install with

```
sudo apt-get install libhdf5-serial-dev netcdf-bin libnetcdf-dev
```

in ubuntu

# Install

`pip install boutdata`

or in a Conda environment:

`conda install -c conda-forge boutdata`

### Examples
Reading data from dump files:

```
from boutdata import collect
ni = collect("Ni")
```

When boutdata is installed as a package (e.g. with pip or conda), a
command-line program is provided as an interface to squashoutput:
```
$ bout-squashoutput
```
bout-squashoutput provides tab-completion of arguments if argcomplete is
installed and activated (see
[https://kislyuk.github.io/argcomplete/](https://kislyuk.github.io/argcomplete/)).
