"""Routines for redistributing files over different numbers of
processors

"""

from collections import namedtuple
from math import sqrt

processor_layout_ = namedtuple(
    "BOUT_processor_layout",
    [
        "nxpe",
        "nype",
        "nzpe",
        "npes",
        "mxsub",
        "mysub",
        "mzsub",
        "nx",
        "ny",
        "nz",
        "mxg",
        "myg",
        "mzg",
    ],
)


# Subclass the namedtuple above so we can add a docstring
class processor_layout(processor_layout_):
    """A namedtuple describing the processor layout, including grid sizes
    and guard cells

    Parameters
    ----------

    nxpe, nype : int
        The number of processors in x and y
    npes : int
        The total number of procesors
    mxsub, mysub : int
        The size of the grid in x and y on a single processor
    nx, ny, mz : int
        The total size of the grid in x, y and z
    mxg : int
        The number of guard cells in x and y

    """

    pass


def get_processor_layout(boutfile, has_t_dimension=True, mxg=None, myg=None, mzg=None):
    """Given a BOUT.restart.* or BOUT.dmp.* file (as a DataFile object),
    return the processor layout for its data

    Parameters
    ----------
    boutfile : DataFile
        Restart or dump file to read
    has_t_dimension : bool, optional
        Does this file have a time dimension?
    mxg, myg : int, optional
        Number of x, y guard cells

    Returns
    -------
    processor_layout
        A description of the processor layout and grid sizes

    """

    mxg = mxg or boutfile.get("MXG", 2)
    myg = myg or boutfile.get("MYG", 2)
    mzg = mzg or boutfile.get("MZG", 2)

    nxpe = boutfile.read("NXPE")
    nype = boutfile.read("NYPE")
    nzpe = boutfile.read("NZPE")
    npes = nxpe * nype * nzpe

    # Get list of variables
    var_list = boutfile.list()
    if len(var_list) == 0:
        raise ValueError("ERROR: No data found")

    mxsub = 0
    mysub = 0
    mzsub = 0

    if has_t_dimension:
        maxdims = 4
    else:
        maxdims = 3
    for v in var_list:
        if boutfile.ndims(v) == maxdims:
            s = boutfile.size(v)
            mxsub = s[maxdims - 3] - 2 * mxg
            if mxsub < 0:
                if s[maxdims - 3] == 1:
                    mxsub = 1
                    mxg = 0
                elif s[maxdims - 3] == 3:
                    mxsub = 1
                    mxg = 1
                else:
                    print("Number of x points is wrong?")
                    return False

            mysub = s[maxdims - 2] - 2 * myg
            if mysub < 0:
                if s[maxdims - 2] == 1:
                    mysub = 1
                    myg = 0
                elif s[maxdims - 2] == 3:
                    mysub = 1
                    myg = 1
                else:
                    print("Number of y points is wrong?")
                    return False

            mzsub = s[maxdims - 1] - 2 * mzg
            if mzsub < 0:
                if s[maxdims - 2] == 1:
                    mzsub = 1
                    mzg = 0
                elif s[maxdims - 2] == 3:
                    mzsub = 1
                    mzg = 1
                else:
                    print("Number of z points is wrong?")
                    return False
            break

    # Calculate total size of the grid
    nx = mxsub * nxpe
    ny = mysub * nype
    nz = mzsub * nzpe

    result = processor_layout(
        nxpe=nxpe,
        nype=nype,
        nzpe=nzpe,
        npes=npes,
        mxsub=mxsub,
        mysub=mysub,
        mzsub=mzsub,
        nx=nx,
        ny=ny,
        nz=nz,
        mxg=mxg,
        myg=myg,
        mzg=mzg,
    )

    return result


def create_processor_layout(old_processor_layout, npes, nxpe=None, nzpe=1):
    """Convert one processor layout into another one with a different
    total number of processors

    If nxpe is None, use algorithm from BoutMesh to select optimal nxpe.
    Otherwise, check nxpe is valid (divides npes)

    Parameters
    ----------
    old_processor_layout : processor_layout
        The processor layout to convert
    npes : int
        The new total number of procesors
    nxpe : int, optional
        The number of procesors in x to use

    Returns
    -------
    processor_layout
        A description of the processor layout and grid sizes

    """

    npes = npes // nzpe

    if nxpe is None:  # Copy algorithm from BoutMesh for selecting nxpe
        ideal = sqrt(
            float(old_processor_layout.nx)
            * float(npes)
            / float(old_processor_layout.ny)
        )
        # Results in square domain

        for i in range(1, npes + 1):
            if (
                npes % i == 0
                and old_processor_layout.nx % i == 0
                and int(old_processor_layout.nx / i) >= old_processor_layout.mxg
                and old_processor_layout.ny % (npes / i) == 0
            ):
                # Found an acceptable value
                # Warning: does not check branch cuts!

                if nxpe is None or abs(ideal - i) < abs(ideal - nxpe):
                    nxpe = i  # Keep value nearest to the ideal

        if nxpe is None:
            raise ValueError("ERROR: could not find a valid value for nxpe")
    elif npes % nxpe != 0:
        raise ValueError("ERROR: requested nxpe is invalid, it does not divide npes")

    nype = int(npes / nxpe)

    mxsub = int(old_processor_layout.nx / nxpe)
    mysub = int(old_processor_layout.ny / nype)
    mzsub = int(old_processor_layout.nz / nzpe)

    result = processor_layout(
        nxpe=nxpe,
        nype=nype,
        nzpe=nzpe,
        npes=npes,
        mxsub=mxsub,
        mysub=mysub,
        mzsub=mzsub,
        nx=old_processor_layout.nx,
        ny=old_processor_layout.ny,
        nz=old_processor_layout.nz,
        mxg=old_processor_layout.mxg,
        myg=old_processor_layout.myg,
        mzg=old_processor_layout.mzg,
    )

    return result
