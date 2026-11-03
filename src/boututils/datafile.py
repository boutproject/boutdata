"""File I/O class

A wrapper around various NetCDF libraries and h5py, used by BOUT++
routines. Creates a consistent interface across machines

Supported libraries:

- ``h5py`` (for HDF5 files)
- ``netCDF4`` (preferred NetCDF library)

NetCDF-4 files are HDF5 files. When ``h5py`` is available, NetCDF-4 files
opened read-only are read with ``h5py``, which opens files much faster than
``netCDF4``. The reader can be chosen with the environment variable
``BOUTDATA_NETCDF_READER`` (or by setting ``boututils.datafile.netcdf_reader``):

- ``"auto"`` (default): use ``h5py`` if it is installed, otherwise ``netCDF4``
- ``"netcdf4"``: always use ``netCDF4``
- ``"h5py"``: always use ``h5py`` for NetCDF-4 files opened read-only

NOTE
----
NetCDF and HDF5 include unlimited dimensions, but this library is just
for very simple I/O operations. Educated guesses are made for the
dimensions.

TODO
----
- Don't raise ``ImportError`` if no NetCDF libraries found, use HDF5
  instead?
- Cleaner handling of different NetCDF libraries
- Support for h5netcdf?

"""

import os
import pathlib

import numpy as np

from boututils.boutarray import BoutArray
from boututils.boutwarnings import alwayswarn

try:
    from netCDF4 import Dataset, default_fillvals

    has_netCDF = True
except ImportError:
    raise ImportError(
        "DataFile: No supported NetCDF modules available -- requires netCDF4"
    )

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False

# Which library to use to read NetCDF-4 files: "auto", "netcdf4" or "h5py"
netcdf_reader = os.environ.get("BOUTDATA_NETCDF_READER", "auto")


def _use_h5py_reader(filename, write, create):
    """Should the NetCDF file `filename` be read with h5py?"""
    if netcdf_reader not in ("auto", "netcdf4", "h5py"):
        raise ValueError(
            f"Invalid netcdf_reader '{netcdf_reader}'. Should be 'auto', 'netcdf4' "
            "or 'h5py'"
        )
    if write or create or netcdf_reader == "netcdf4":
        return False
    if netcdf_reader == "h5py" and not has_h5py:
        raise ImportError("netcdf_reader is 'h5py' but h5py is not available")
    if not has_h5py or filename.suffix not in (".nc", ".ncdf"):
        return False
    # NetCDF-3 files are not HDF5 files, so need to be read with netCDF4
    return h5py.is_hdf5(filename)


class DataFile:
    """File I/O class

    A wrapper around various NetCDF libraries and h5py, used by BOUT++
    routines. Creates a consistent interface across machines

    Parameters
    ----------
    filename : str, optional
        Name of file to open. If no filename supplied, you will need
        to call :py:obj:`~DataFile.open` and supply `filename` there
    write : bool, optional
        If True, open the file in read-write mode (existing files will
        be appended to). Default is read-only mode
    create : bool, optional
        If True, open the file in write mode (existing files will be
        truncated). Default is read-only mode
    format : str, optional
        Name of a filetype to use (e.g. ``NETCDF3_CLASSIC``,
        ``NETCDF3_64BIT``, ``NETCDF4``, ``HDF5``)

    TODO
    ----
    - `filename` should not be optional!
    - Take a ``mode`` argument to be more in line with other file types
    - `format` should be checked to be a sensible value
    - Make sure ``__init__`` methods are first
    - Make `impl` and `handle` private

    """

    impl = None

    def __init__(
        self, filename=None, write=False, create=False, format="NETCDF4", **kwargs
    ):
        """

        NetCDF formats are described here: https://unidata.github.io/netcdf4-python/
        - NETCDF3_CLASSIC   Limited to 2.1Gb files
        - NETCDF3_64BIT_OFFSET or NETCDF3_64BIT is an extension to allow larger files
        - NETCDF3_64BIT_DATA adds 64-bit integer data types and 64-bit dimension sizes
        - NETCDF4 and NETCDF4_CLASSIC use HDF5 as the disk format
        """
        if filename is not None:
            filename = pathlib.Path(filename)
            if filename.suffix in (".hdf5", ".hdf", ".h5"):
                self.impl = DataFile_HDF5(
                    filename=filename, write=write, create=create, format=format
                )
            elif _use_h5py_reader(filename, write, create):
                self.impl = DataFile_netCDF_h5py(filename=filename)
            else:
                self.impl = DataFile_netCDF(
                    filename=filename,
                    write=write,
                    create=create,
                    format=format,
                    **kwargs,
                )
        elif format == "HDF5":
            self.impl = DataFile_HDF5(
                filename=filename, write=write, create=create, format=format
            )
        else:
            self.impl = DataFile_netCDF(
                filename=filename, write=write, create=create, format=format, **kwargs
            )

    def open(self, filename, write=False, create=False, format="NETCDF3_CLASSIC"):
        """Open the file

        Parameters
        ----------
        filename : str, optional
            Name of file to open
        write : bool, optional
            If True, open the file in read-write mode (existing files will
            be appended to). Default is read-only mode
        create : bool, optional
            If True, open the file in write mode (existing files will be
            truncated). Default is read-only mode
        format : str, optional
            Name of a filetype to use (e.g. ``NETCDF3_CLASSIC``,
            ``NETCDF4``, ``HDF5``)

        TODO
        ----
        - Return the result of calling open to be more like stdlib's
          open
        - `keys` should be more pythonic (return generator)

        """
        self.impl.open(filename, write=write, create=create, format=format)

    def close(self):
        """Close a file and flush data to disk"""
        self.impl.close()

    def __del__(self):
        if self.impl is not None:
            self.impl.__del__()

    def __enter__(self):
        self.impl.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.impl.__exit__(type, value, traceback)

    def read(self, name, ranges=None, asBoutArray=True):
        """Read a variable from the file

        Parameters
        ----------
        name : str
            Name of the variable to read
        ranges : list of slice objects, optional
            Slices of variable to read, can also be converted from lists or
            tuples of (start, stop, stride). The number of elements in `ranges`
            should be equal to the number of dimensions of the variable you
            wish to read. See :py:obj:`~DataFile.size` for how to get the
            dimensions
        asBoutArray : bool, optional
            If True, return the variable as a
            :py:obj:`~boututils.boutarray.BoutArray` (the default)

        Returns
        -------
        ndarray or :py:obj:`~boututils.boutarray.BoutArray`
            The variable from the file
            (:py:obj:`~boututils.boutarray.BoutArray` if `asBoutArray`
            is True)

        """
        if ranges is not None:
            for x in ranges:
                if isinstance(x, list | tuple):
                    x = slice(*x)
        return self.impl.read(name, ranges=ranges, asBoutArray=asBoutArray)

    def list(self):
        """List all variables in the file

        Returns
        -------
        list of str
            A list containing all the names of the variables

        """
        return self.impl.list()

    def keys(self):
        """A synonym for :py:obj:`~DataFile.list`

        TODO
        ----
        - Make a generator to be more like python3 dict keys

        """
        return self.list()

    def dimensions(self, varname):
        """Return the names of all the dimensions of a variable

        Parameters
        ----------
        varname : str
            The name of the variable

        Returns
        -------
        tuple of str
            The names of the variable's dimensions

        """
        return self.impl.dimensions(varname)

    def ndims(self, varname):
        """Return the number of dimensions for a variable

        Parameters
        ----------
        varname : str
            The name of the variable

        Returns
        -------
        int
            The number of dimensions

        """
        return self.impl.ndims(varname)

    def sync(self):
        """Write pending changes to disk."""
        self.impl.sync()

    def size(self, varname):
        """Return the size of each dimension of a variable

        Parameters
        ----------
        varname : str
            The name of the variable

        Returns
        -------
        tuple of int
            The size of each dimension

        """
        return self.impl.size(varname)

    def bout_type(self, varname):
        """Return the name of the BOUT++ type of a variable

        Possible values are:

        - scalar
        - Field2D
        - Field3D

        If the variable is an evolving variable (i.e. has a time
        dimension), then it is appended with a "_t"

        Parameters
        ----------
        varname : str
            The name of the variable

        Returns
        -------
        str
            The name of the BOUT++ type

        """
        return self.attributes(varname)["bout_type"]

    def write(self, name, data, info=False, *, dims=None):
        """Write a variable to file

        If the variable is not a :py:obj:`~boututils.boutarray.BoutArray` with
        the ``bout_type`` attribute, a guess will be made for the
        dimensions

        Parameters
        ----------
        name : str
            Name of the variable to use in the file
        data : :py:obj:`~boututils.boutarray.BoutArray` or ndarray
            An array containing the variable data
        info : bool, optional
            If True, print information about what is being written to
            file
        dims : tuple(str) or None, optional
            If passed, specifies the dimensions to be used to write the variable.

        Returns
        -------
        None

        """
        return self.impl.write(name, data, info, dims=dims)

    def read_file_attribute(self, name):
        return self.impl.read_file_attribute(name)

    def write_file_attribute(self, name, value):
        return self.impl.write_file_attribute(name, value)

    def list_file_attributes(self):
        return self.impl.list_file_attributes()

    def get(self, name, default=None):
        if default is None or name in self.keys():
            return self[name]
        else:
            return default

    def __getitem__(self, name):
        return self.impl.__getitem__(name)

    def __setitem__(self, key, value):
        self.impl.__setitem__(key, value)

    def attributes(self, varname):
        """Return a dictionary of attributes

        Parameters
        ----------
        varname : str
            The name of the variable

        Returns
        -------
        dict
            The attribute names and their values

        """
        return self.impl.attributes(varname)


class DataFile_netCDF(DataFile):
    handle = None

    def open(self, filename, write=False, create=False, format="NETCDF3_CLASSIC"):
        if (not write) and (not create):
            self.handle = Dataset(filename, "r")
        elif create:
            self.handle = Dataset(filename, "w", format=format)
        else:
            self.handle = Dataset(filename, "a")
        # Record if writing
        self.writeable = write or create

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    def __init__(
        self,
        filename=None,
        write=False,
        create=False,
        format="NETCDF3_CLASSIC",
        **kwargs,
    ):
        self._kwargs = kwargs
        if not has_netCDF:
            message = "DataFile: No supported NetCDF python-modules available"
            raise ImportError(message)
        if filename is not None:
            self.open(filename, write=write, create=create, format=format)
        self._attributes_cache = {}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def read(self, name, ranges=None, asBoutArray=True):
        """Read a variable from the file."""
        if self.handle is None:
            return None

        try:
            var = self.handle.variables[name]
            n = name
        except KeyError:
            # Not found. Try to find using case-insensitive search
            var = None
            for n in list(self.handle.variables.keys()):
                if n.lower() == name.lower():
                    print(f"WARNING: Reading '{n}' instead of '{name}'")
                    var = self.handle.variables[n]
            if var is None:
                return None

        if asBoutArray:
            attributes = self.attributes(n)

        ndims = len(var.dimensions)
        if ndims == 0:
            data = var.getValue()
            if asBoutArray:
                data = BoutArray(data, attributes=attributes)
            return data  # [0]
        else:
            if ranges:
                if len(ranges) == 2 * ndims:
                    # Reform list of pairs of ints into slices
                    ranges = [slice(a, b) for a, b in zip(ranges[::2], ranges[1::2])]
                elif len(ranges) != ndims:
                    raise ValueError(
                        "Incorrect number of elements in ranges argument "
                        f"(got {ranges}, expected {ndims} or {2 * ndims})"
                    )

                data = var[ranges[:ndims]]
                if asBoutArray:
                    data = BoutArray(data, attributes=attributes)
                return data
            else:
                data = var[:]
                if asBoutArray:
                    data = BoutArray(data, attributes=attributes)
                return data

    def __getitem__(self, name):
        var = self.read(name)
        if var is None:
            raise KeyError(f"No variable found: {name}")
        return var

    def __setitem__(self, key, value):
        self.write(key, value)

    def list(self):
        if self.handle is None:
            return []
        return list(self.handle.variables.keys())

    def keys(self):
        return self.list()

    def dimensions(self, varname):
        if self.handle is None:
            return None
        try:
            var = self.handle.variables[varname]
        except KeyError:
            raise ValueError("No such variable")
        return var.dimensions

    def ndims(self, varname):
        if self.handle is None:
            raise ValueError("File not open")
        try:
            var = self.handle.variables[varname]
        except KeyError:
            raise ValueError("No such variable")
        return len(var.dimensions)

    def sync(self):
        self.handle.sync()

    def size(self, varname):
        if self.handle is None:
            return []
        try:
            var = self.handle.variables[varname]
        except KeyError:
            return []

        def dimlen(d):
            dim = self.handle.dimensions[d]
            if dim is not None:
                t = type(dim).__name__
                if t == "int":
                    return dim
                return len(dim)
            return 0

        return [dimlen(d) for d in var.dimensions]

    def _bout_type_from_dimensions(self, varname):
        dims = self.dimensions(varname)

        if any("char" in d for d in dims):
            if "t" in dims:
                return "string_t"
            else:
                return "string"

        return BoutArray.type_from_dims(dims)

    def _bout_dimensions_from_var(self, data):
        try:
            bout_type = data.attributes["bout_type"]
        except AttributeError:
            if hasattr(data, "dims"):
                return data.dims
            defdims_list = [
                (),
                ("t",),
                ("x", "y"),
                ("x", "y", "z"),
                ("t", "x", "y", "z"),
            ]
            return defdims_list[len(np.shape(data))]

        if bout_type == "string_t":
            _, string_length = data.shape
            return ("t", f"char{string_length}")
        elif bout_type == "string":
            return (f"char{len(data)}",)

        return BoutArray.dims_from_type(bout_type)

    def write(self, name, data, info=False, *, dims=None):
        if not self.writeable:
            raise Exception("File not writeable. Open with write=True keyword")

        s = np.shape(data)

        if dims is not None and len(dims) != data.ndim:
            raise ValueError(
                f"When dims is passed, it must have an entry for each dimension of "
                f"`data`, but dims={dims} and data.ndim={data.ndim}."
            )

        # Get the variable type
        t = type(data).__name__

        if t == "DataArray":
            t = data.dtype.str

        if t == "NoneType":
            print("DataFile: None passed as data to write. Ignoring")
            return

        if t == "ndarray" or t == "BoutArray":
            # Numpy type or BoutArray wrapper for Numpy type. Get the data type
            t = data.dtype.str

        if t == "list":
            # List -> convert to numpy array
            data = np.array(data)
            t = data.dtype.str

        if (t == "int") or (t == "<i8") or (t == "int64"):
            # NetCDF 3 does not support type int64
            data = np.int32(data)
            t = data.dtype.str

        try:
            # See if the variable already exists
            var = self.handle.variables[name]

            # Check the shape of the variable
            if var.shape != s:
                print(f"DataFile: Variable already exists with different size: {name}")
                # Fallthrough to the exception
                raise KeyError
        except KeyError:
            # Not found, so add.

            # Get dimensions
            if dims is None:
                defdims = self._bout_dimensions_from_var(data)
            else:
                defdims = dims

            def find_dim(dim):
                # Find a dimension with given name and size
                size, name = dim

                # See if it exists already
                try:
                    d = self.handle.dimensions[name]

                    # Check if it's the correct size
                    if type(d).__name__ == "int":
                        if d == size:
                            return name
                    else:
                        if len(d) == size:
                            return name

                    # Find another with the correct size
                    for dn, d in list(self.handle.dimensions.items()):
                        # Some implementations need len(d) here, some just d
                        if type(d).__name__ == "int":
                            if d == size:
                                return dn
                        else:
                            if len(d) == size:
                                return dn

                    # None found, so create a new one
                    i = 2
                    while True:
                        dn = f"{name}{i}"
                        try:
                            d = self.handle.dimensions[dn]
                            # Already exists, so keep going
                        except KeyError:
                            # Not found. Create
                            if info:
                                print(f"Defining dimension {dn} of size {size}")

                            self.handle.createDimension(dn, size)
                            return dn
                        i = i + 1

                except KeyError:
                    # Doesn't exist, so add
                    if info:
                        print(f"Defining dimension {name} of size {size}")
                    if name == "t":
                        size = None

                    self.handle.createDimension(name, size)

                return name

            # List of (size, 'name') tuples
            dlist = list(zip(s, defdims))
            # Get new list of variables, and turn into a tuple
            dims_tuple = tuple(map(find_dim, dlist))

            # Create the variable
            var = self.handle.createVariable(name, t, dims_tuple, **self._kwargs)

            if var is None:
                raise Exception("Couldn't create variable")

        # Write the data
        if t == "str":
            var[0] = data
        else:
            try:
                # Some libraries allow this for arrays
                var.assignValue(data)
            except Exception:
                # And some others only this
                var[:] = data

        # Write attributes, if present
        try:
            for attrname, attrval in data.attributes.items():
                if isinstance(attrval, int):
                    attrval = np.int32(attrval)
                var.setncattr(attrname, attrval)
        except AttributeError:
            pass

    def read_file_attribute(self, name):
        try:
            return getattr(self.handle, name)
        except AttributeError:
            raise AttributeError(f"DataFile (netCDF4) has no file attribute {name}")

    def write_file_attribute(self, name, value):
        setattr(self.handle, name, value)

    def list_file_attributes(self):
        return self.handle.ncattrs()

    def attributes(self, varname):
        try:
            return self._attributes_cache[varname]
        except KeyError:
            # Need to build the attributes dictionary for this variable
            if self.handle is None:
                return None
            try:
                var = self.handle.variables[varname]
            except KeyError:
                # Not found. Try to find using case-insensitive search
                var = None
                for n in list(self.handle.variables.keys()):
                    if n.lower() == varname.lower():
                        print(f"WARNING: Reading '{n}' instead of '{varname}'")
                        var = self.handle.variables[n]
                if var is None:
                    return None

            attributes = {}  # Map of attribute names to values

            try:
                attribs = var.ncattrs()  # List of attributes
                for attrname in attribs:
                    attributes[attrname] = var.getncattr(
                        attrname
                    )  # Get all values and insert into map
            except Exception:
                print(f"Error reading attributes for {varname}")
                # Result will be an empty map

            if "bout_type" not in attributes:
                attributes["bout_type"] = self._bout_type_from_dimensions(varname)

            # Save the attributes for this variable to the cache
            self._attributes_cache[varname] = attributes

            return attributes


class DataFile_netCDF_h5py(DataFile_netCDF):
    """Read-only access to NetCDF-4 files using h5py

    NetCDF-4 files are HDF5 files, and h5py opens them much faster than netCDF4,
    which reads the metadata of all variables when opening a file. Results are the
    same as from DataFile_netCDF. Variables that h5py cannot read in the same way
    as netCDF4 (e.g. strings arrays or packed data) are read using DataFile_netCDF.
    """

    # Attributes used internally by NetCDF-4 and HDF5 dimension scales, which netCDF4
    # does not show
    _hidden_attributes = (
        "_Netcdf4Coordinates",
        "_Netcdf4Dimid",
        "_NCProperties",
        "_IsNetcdf4",
        "_SuperblockVersion",
        "_nc3_strict",
        "DIMENSION_LIST",
        "REFERENCE_LIST",
        "CLASS",
        "NAME",
    )

    # Attributes that make netCDF4 modify the data it reads
    _netcdf4_conversion_attributes = (
        "scale_factor",
        "add_offset",
        "missing_value",
        "valid_min",
        "valid_max",
        "valid_range",
        "_Unsigned",
        "_Encoding",
    )

    _netcdf4_file = None

    # Metadata (list of variables, dimensions, attributes) of files that have been
    # opened, so that it only needs to be read once, rather than every time a file is
    # opened. Keys are (path, modification time, size, inode), so the metadata is read
    # again if a file changes.
    _metadata_cache = {}  # noqa: RUF012
    _metadata_cache_max_files = 10000

    def open(self, filename, write=False, create=False, format=None):
        if write or create:
            raise ValueError("DataFile_netCDF_h5py can only read files")
        self.handle = h5py.File(filename, mode="r")
        self.writeable = False

        stat = os.stat(filename)
        cache_key = (
            os.path.abspath(filename),
            stat.st_mtime_ns,
            stat.st_size,
            stat.st_ino,
        )
        if len(self._metadata_cache) >= self._metadata_cache_max_files:
            self._metadata_cache.clear()
        self._metadata = self._metadata_cache.setdefault(
            cache_key,
            {
                "contents": None,
                "attributes": {},
                "needs_netcdf4": {},
                "dimension_lengths": {},
            },
        )

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        if self._netcdf4_file is not None:
            self._netcdf4_file.close()
        self._netcdf4_file = None

    def __init__(self, filename=None, write=False, create=False, format=None):
        if not has_h5py:
            message = "DataFile: No supported HDF5 python-modules available"
            raise ImportError(message)
        self._filename = filename
        self._kwargs = {}
        if filename is not None:
            self.open(filename, write=write, create=create, format=format)
        self._attributes_cache = {}

    def _get_netcdf4_file(self):
        """DataFile_netCDF for the same file, for variables h5py cannot read"""
        if self._netcdf4_file is None:
            self._netcdf4_file = DataFile_netCDF(self._filename)
        return self._netcdf4_file

    def _read_name_attribute(self, name):
        """Read the "NAME" attribute of dimension scale `name`, or None if not a scale

        Uses the low-level h5py API, because this is much faster than opening the
        dataset.
        """
        name = name.encode()
        try:
            if not h5py.h5a.exists(self.handle.id, b"NAME", obj_name=name):
                return None
        except (KeyError, RuntimeError):
            # No object called `name`
            return None
        attribute = h5py.h5a.open(self.handle.id, b"NAME", obj_name=name)
        value = np.empty((), dtype=attribute.dtype)
        attribute.read(value)
        return bytes(value)

    def _is_dimension_only(self, name):
        """Is `name` a NetCDF dimension that is not also a variable?"""
        scale_name = self._read_name_attribute(name)
        return scale_name is not None and scale_name.startswith(
            b"This is a netCDF dimension but not"
        )

    def _get_variable(self, name):
        """Get the h5py.Dataset of a variable, or None if there is no such variable"""
        if self.handle is None or "/" in name or name.startswith("_nc4_non_coord_"):
            return None
        if self._is_dimension_only(name):
            # A variable with the same name as a dimension, but different
            # dimensions, is renamed by NetCDF
            name = "_nc4_non_coord_" + name
        try:
            var = self.handle[name]
        except KeyError:
            return None
        if not isinstance(var, h5py.Dataset):
            return None
        return var

    def _needs_netcdf4(self, name, var):
        """Does variable `var` have to be read with netCDF4 to get the same result?"""
        needs_netcdf4 = self._metadata["needs_netcdf4"]
        if name not in needs_netcdf4:
            needs_netcdf4[name] = self._check_needs_netcdf4(var)
        return needs_netcdf4[name]

    def _check_needs_netcdf4(self, var):
        if var.dtype.kind not in "iuf":
            # Only scalar strings are handled here
            return not (var.shape == () and h5py.check_string_dtype(var.dtype))
        if not var.dtype.isnative:
            # Leave byte-swapping to netCDF4
            return True
        if (
            var.shape != ()
            and "DIMENSION_LIST" not in var.attrs
            and (var.attrs.get("CLASS") != b"DIMENSION_SCALE" or len(var.shape) != 1)
        ):
            # Not a NetCDF-4 variable, and not a 1d coordinate variable
            return True
        if any(a in var.attrs for a in self._netcdf4_conversion_attributes):
            return True
        # netCDF4 pads variables that are shorter than an unlimited dimension with
        # fill values
        return any(
            maxsize is None and size < length
            for size, maxsize, length in zip(
                var.shape, var.maxshape, self._variable_dimension_lengths(var)
            )
        )

    @staticmethod
    def _convert_attribute(value):
        """Convert an attribute read by h5py to the value netCDF4 would give"""
        if isinstance(value, h5py.Empty):
            return ""
        if isinstance(value, bytes | np.bytes_):
            return value.decode("utf-8", "replace").replace("\x00", "")
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "OSU":
                strings = [DataFile_netCDF_h5py._convert_attribute(v) for v in value]
                return strings[0] if len(strings) == 1 else strings
            if value.shape == (1,):
                return value[0]
        return value

    def _mask_fill_values(self, var, data):
        """Mask fill values in the same way as netCDF4"""
        fill_value = var.attrs.get("_FillValue")
        if fill_value is None:
            no_fill = (
                var.id.get_create_plist().get_fill_time() == h5py.h5d.FILL_TIME_NEVER
            )
            if no_fill or var.dtype.str[1:] in ("i1", "u1"):
                return np.ma.masked_array(data)
            fill_value = default_fillvals[var.dtype.str[1:]]
        if np.isnan(fill_value):
            return np.ma.masked_array(data, mask=np.isnan(data))
        return np.ma.masked_array(data, mask=(data == fill_value))

    def read(self, name, ranges=None, asBoutArray=True):
        """Read a variable from the file."""
        if self.handle is None:
            return None

        var = self._get_variable(name)
        n = name
        if var is None:
            # Not found. Try to find using case-insensitive search
            for n in self.list():
                if n.lower() == name.lower():
                    print(f"WARNING: Reading '{n}' instead of '{name}'")
                    var = self._get_variable(n)
            if var is None:
                return None

        if self._needs_netcdf4(n, var):
            return self._get_netcdf4_file().read(n, ranges, asBoutArray)

        if asBoutArray:
            attributes = self.attributes(n)

        ndims = len(var.shape)
        if ndims == 0:
            if h5py.check_string_dtype(var.dtype):
                data = var.asstr()[()]
            else:
                data = np.asarray(var[()])
                if not asBoutArray:
                    data = self._mask_fill_values(var, data)
            if asBoutArray:
                data = BoutArray(data, attributes=attributes)
            return data

        if ranges:
            if len(ranges) == 2 * ndims:
                # Reform list of pairs of ints into slices
                ranges = [slice(a, b) for a, b in zip(ranges[::2], ranges[1::2])]
            elif len(ranges) != ndims:
                raise ValueError(
                    "Incorrect number of elements in ranges argument "
                    f"(got {ranges}, expected {ndims} or {2 * ndims})"
                )
            ranges = ranges[:ndims]
            if any(
                r is None or (isinstance(r, slice) and (r.step or 1) < 0)
                for r in ranges
            ):
                # Indexing that h5py does not support in the same way as netCDF4
                return self._get_netcdf4_file().read(n, ranges, asBoutArray)
            data = var[tuple(ranges)]
        else:
            data = var[...]

        if asBoutArray:
            return BoutArray(data, attributes=attributes)
        return self._mask_fill_values(var, data)

    def sync(self):
        self.handle.flush()

    def _get_contents(self):
        """Scan the file for its variables and dimensions

        Returns
        -------
        variables : list of str
            Names of the variables, in the same order as netCDF4 lists them
        dimension_names : dict
            Map from NetCDF dimension IDs to dimension names
        """
        if self._metadata["contents"] is not None:
            return self._metadata["contents"]

        datasets = []

        def add_dataset(name, info):
            # visit() is recursive, so skip objects inside groups
            if info.type == h5py.h5o.TYPE_DATASET and b"/" not in name:
                datasets.append(name.decode())

        # Use the creation order of the variables, like netCDF4, if it was saved
        try:
            h5py.h5o.visit(
                self.handle.id, add_dataset, info=True, idx_type=h5py.h5.INDEX_CRT_ORDER
            )
        except (KeyError, RuntimeError, ValueError):
            datasets = []
            h5py.h5o.visit(
                self.handle.id, add_dataset, info=True, idx_type=h5py.h5.INDEX_NAME
            )

        variables = []
        dimension_names = {}
        for name in datasets:
            scale_name = self._read_name_attribute(name)
            if scale_name is not None:
                # A dimension scale, i.e. a NetCDF dimension
                dimension_id = self.handle[name].attrs.get("_Netcdf4Dimid")
                if dimension_id is not None:
                    dimension_names[int(dimension_id)] = name
                if scale_name.startswith(b"This is a netCDF dimension but not"):
                    continue
            variables.append(name.removeprefix("_nc4_non_coord_"))

        self._metadata["contents"] = variables, dimension_names
        return self._metadata["contents"]

    def list(self):
        if self.handle is None:
            return []
        return list(self._get_contents()[0])

    def dimensions(self, varname):
        if self.handle is None:
            return None
        var = self._get_variable(varname)
        if var is None:
            raise ValueError("No such variable")
        if self._needs_netcdf4(varname, var):
            return self._get_netcdf4_file().dimensions(varname)
        if var.shape == ():
            return ()
        if "DIMENSION_LIST" not in var.attrs:
            # 1d coordinate variable: the variable is its own dimension
            return (varname,)
        dimension_names = self._get_contents()[1]
        dimension_ids = var.attrs.get("_Netcdf4Coordinates")
        if dimension_ids is not None and all(
            i in dimension_names for i in dimension_ids
        ):
            return tuple(dimension_names[i] for i in dimension_ids)
        # Look up the names of the dimension scales. This is slow, because HDF5 has to
        # search the file for the name of each referenced object
        return tuple(
            self.handle[refs[0]].name.removeprefix("/")
            for refs in var.attrs["DIMENSION_LIST"]
        )

    def ndims(self, varname):
        if self.handle is None:
            raise ValueError("File not open")
        return len(self.dimensions(varname))

    def size(self, varname):
        if self.handle is None:
            return []
        var = self._get_variable(varname)
        if var is None:
            return []
        if self._needs_netcdf4(varname, var):
            return self._get_netcdf4_file().size(varname)
        # Use the size of the dimensions, like netCDF4
        return self._variable_dimension_lengths(var)

    def _variable_dimension_lengths(self, var):
        """Lengths of the dimensions of `var`, as NetCDF defines them"""
        if var.shape == ():
            return []
        if "DIMENSION_LIST" not in var.attrs:
            # 1d coordinate variable, which is its own dimension
            return [self._dimension_length(var)]
        return [
            self._dimension_length(self.handle[refs[0]])
            for refs in var.attrs["DIMENSION_LIST"]
        ]

    def _dimension_length(self, dimension):
        """Length of a dimension, given its dimension scale dataset"""
        if dimension.maxshape[0] is not None:
            return dimension.shape[0]

        # Unlimited dimension: NetCDF uses the longest variable that uses it
        dimension_lengths = self._metadata["dimension_lengths"]
        dimension_id = int(dimension.attrs.get("_Netcdf4Dimid", -1))
        if dimension_id in dimension_lengths:
            return dimension_lengths[dimension_id]
        length = dimension.shape[0]
        if "REFERENCE_LIST" in dimension.attrs:
            for ref, axis in dimension.attrs["REFERENCE_LIST"]:
                length = max(length, self.handle[ref].shape[axis])
        if dimension_id >= 0:
            dimension_lengths[dimension_id] = length
        return length

    def write(self, name, data, info=False, *, dims=None):
        raise Exception("File not writeable. Open with write=True keyword")

    def read_file_attribute(self, name):
        if name in self._hidden_attributes or name not in self.handle.attrs:
            raise AttributeError(f"DataFile (netCDF4) has no file attribute {name}")
        return self._convert_attribute(self.handle.attrs[name])

    def write_file_attribute(self, name, value):
        raise Exception("File not writeable. Open with write=True keyword")

    def list_file_attributes(self):
        return [a for a in self.handle.attrs if a not in self._hidden_attributes]

    def attributes(self, varname):
        try:
            return self._attributes_cache[varname]
        except KeyError:
            # Need to build the attributes dictionary for this variable
            if self.handle is None:
                return None
            shared_attributes = self._metadata["attributes"]
            if varname in shared_attributes:
                # Copy, so that changing the result does not change the shared cache
                attributes = dict(shared_attributes[varname])
                self._attributes_cache[varname] = attributes
                return attributes

            var = self._get_variable(varname)
            n = varname
            if var is None:
                # Not found. Try to find using case-insensitive search
                for n in self.list():
                    if n.lower() == varname.lower():
                        print(f"WARNING: Reading '{n}' instead of '{varname}'")
                        var = self._get_variable(n)
                if var is None:
                    return None

            if self._needs_netcdf4(n, var):
                attributes = self._get_netcdf4_file().attributes(varname)
            else:
                attributes = {
                    attrname: self._convert_attribute(value)
                    for attrname, value in var.attrs.items()
                    if attrname not in self._hidden_attributes
                }
                if "bout_type" not in attributes:
                    attributes["bout_type"] = self._bout_type_from_dimensions(varname)

            # Save the attributes for this variable to the caches
            self._attributes_cache[varname] = attributes
            if n == varname:
                shared_attributes[varname] = dict(attributes)

            return attributes


class DataFile_HDF5(DataFile):
    handle = None

    def open(self, filename, write=False, create=False, format=None):
        if (not write) and (not create):
            self.handle = h5py.File(filename, mode="r")
        elif create:
            self.handle = h5py.File(filename, mode="w")
        else:
            self.handle = h5py.File(filename, mode="a")
        # Record if writing
        self.writeable = write or create

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    def __init__(self, filename=None, write=False, create=False, format=None):
        if not has_h5py:
            message = "DataFile: No supported HDF5 python-modules available"
            raise ImportError(message)
        if filename is not None:
            self.open(filename, write=write, create=create, format=format)
        self._attributes_cache = {}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def read(self, name, ranges=None, asBoutArray=True):
        if self.handle is None:
            return None

        try:
            var = self.handle[name]
            n = name
        except KeyError:
            # Not found. Try to find using case-insensitive search
            var = None
            for n in self.handle:
                if n.lower() == name.lower():
                    print(f"WARNING: Reading '{n}' instead of '{name}'")
                    var = self.handle[n]
            if var is None:
                return None

        attributes = self.attributes(n) if asBoutArray else {}

        time_dependent = attributes.get("bout_type", "none").endswith("_t")

        ndims = len(var.shape)
        if ndims == 1 and var.shape[0] == 1 and not time_dependent:
            data = var
            if asBoutArray:
                data = BoutArray(data, attributes=attributes)
            return data[0]
        else:
            if ranges:
                if len(ranges) == 2 * ndims:
                    # Reform list of pairs of ints into slices
                    ranges = [slice(a, b) for a, b in zip(ranges[::2], ranges[1::2])]
                elif len(ranges) != ndims:
                    raise ValueError(
                        "Incorrect number of elements in ranges argument "
                        f"(got {ranges}, expected {ndims} or {2 * ndims})"
                    )
                # Probably a bug in h5py, work around by passing tuple
                data = var[tuple(ranges[:ndims])]
                if asBoutArray:
                    data = BoutArray(data, attributes=attributes)
                return data
            else:
                data = var[...]
                if asBoutArray:
                    data = BoutArray(data, attributes=attributes)
                return data

    def __getitem__(self, name):
        var = self.read(name)
        if var is None:
            raise KeyError(f"No variable found: {name}")
        return var

    def __setitem__(self, key, value):
        self.write(key, value)

    def list(self):
        if self.handle is None:
            return []
        names = []
        self.handle.visit(names.append)
        return names

    def keys(self):
        return self.list()

    def dimensions(self, varname):
        bout_type = self.bout_type(varname)
        dims = BoutArray.dims_from_type(bout_type)
        if dims is None:
            raise ValueError(f"Variable bout_type not recognized (got {bout_type})")
        return dims

    def _bout_type_from_array(self, data):
        """Get the bout_type from the array 'data'

        If 'data' is a BoutArray, it knows its bout_type, otherwise we
        have to guess.

        Parameters
        ----------
        data : :py:obj:`~boututils.boutarray.BoutArray` or ndarray
            An array with between 0 and 4 dimensions

        Returns
        -------
        str
            Either the actual bout_type or our best guess

        See Also
        --------
        - `DataFile.bout_type`

        TODO
        ----
        - Make standalone function

        """
        try:
            # If data is a BoutArray, it should have a type attribute that we can use
            return data.attributes["bout_type"]
        except AttributeError:
            # Otherwise data is a numpy.ndarray and we have to guess the bout_type
            pass

        try:
            ndim = len(data.shape)
        except AttributeError:
            ndim = 0
        if ndim == 4:
            return "Field3D_t"
        elif ndim == 3:
            # not ideal, 3d field might be time-evolving 2d field,
            # 'Field2D_t', but can't think of a good way to distinguish
            alwayswarn(
                "Warning: assuming bout_type of 3d array is Field3D. If it "
                "should be a time-evolving Field2D, this may cause errors in "
                "dimension sizes."
            )
            return "Field3D"
        elif ndim == 2:
            return "Field2D"
        elif ndim == 1:
            return "scalar_t"
        elif ndim == 0:
            return "scalar"
        else:
            raise ValueError(f"Unrecognized variable bout_type, ndims={ndim}")

    def ndims(self, varname):
        if self.handle is None:
            return None
        try:
            var = self.handle[varname]
        except KeyError:
            raise ValueError("Variable not found")
        if var.size == 1:
            # variable is a scalar, but h5py always (?) returns numpy.ndarray,
            # so var.shape=(1,)
            return 0
        else:
            return len(var.shape)

    def sync(self):
        self.handle.flush()

    def size(self, varname):
        if self.handle is None:
            return None
        try:
            var = self.handle[varname]
        except KeyError:
            return None
        return var.shape

    def write(self, name, data, info=False, *, dims=None):
        if not self.writeable:
            raise Exception("File not writeable. Open with write=True keyword")

        try:
            bout_type = data.attributes["bout_type"]
        except AttributeError:
            bout_type = self._bout_type_from_array(data)

        if info:
            print(f"Creating variable '{name}' with bout_type '{bout_type}'")

        if bout_type[-2:] == "_t":
            # time evolving fields
            shape = list(data.shape)
            # set time dimension to None to make unlimited
            shape[0] = None
            self.handle.create_dataset(name, data=data, maxshape=shape)
        elif bout_type == "scalar":
            # Need to create scalars as one element arrays to be compatible
            # with BOUT++ assumptions (maybe it would be better to read/write
            # scalars in BOUT++?)
            self.handle.create_dataset(name, data=np.array([data]))
        else:
            self.handle.create_dataset(name, data=data)

        # Need encodes in the following to make sure we pass a byte-string to
        # attrs and not a regular python string.

        # Check if the bout_type of the variable will be written when copying
        # attributes from data, which it should be if data is a BoutArray.
        # Otherwise, need to write it explicitly
        try:
            if "bout_type" not in data.attributes:
                raise AttributeError("'bout_type' not found in attributes")
        except AttributeError:
            self.handle[name].attrs.create(
                "bout_type", bout_type.encode(encoding="utf-8")
            )

        try:
            for attrname, attrval in data.attributes.items():
                if isinstance(attrval, str):
                    attrval = attrval.encode(encoding="utf-8")
                self.handle[name].attrs.create(attrname, attrval)
        except AttributeError:
            # data is not a BoutArray, so doesn't have attributes to write
            pass

    def read_file_attribute(self, name):
        try:
            return self.handle.attrs[name]
        except KeyError:
            raise AttributeError(f"DataFile (HDF5) has no file attribute {name}")

    def write_file_attribute(self, name, value):
        self.handle.attrs[name] = value

    def list_file_attributes(self):
        return self.handle.attrs.keys()

    def attributes(self, varname):
        try:
            return self._attributes_cache[varname]
        except KeyError:
            # Need to add attributes for this variable to the cache
            attributes = {}
            var = self.handle[varname]
            for attrname in var.attrs:
                attribute = var.attrs[attrname]
                if type(attribute) in [bytes, np.bytes_]:
                    attribute = str(attribute, encoding="utf-8")
                attributes[attrname] = attribute

            if "bout_type" not in attributes:
                # bout_type is a required attribute for BOUT++ outputs, so it should
                # have been found
                raise ValueError(
                    f"Error: bout_type not found in attributes of {varname}"
                )

            # Save the attributes for this variable to the cache
            self._attributes_cache[varname] = attributes

            return attributes
