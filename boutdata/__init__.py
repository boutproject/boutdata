""" Routines for exchanging data to/from BOUT++ """

try:
    from builtins import str
except ImportError:
    raise ImportError("Please install the future module to use Python 2")

# Import this, as this almost always used when calling this package
from boutdata.collect import attributes, collect

__all__ = ["attributes", "collect", "gen_surface", "pol_slice"]

__name__ = "boutdata"

try:
    from importlib.metadata import PackageNotFoundError, version
except ModuleNotFoundError:
    from importlib_metadata import PackageNotFoundError, version
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version
    except ModuleNotFoundError as e:
        error_info = (
            "'setuptools_scm' is required to get the version number when running "
            "boutdata from the git repo. Please install 'setuptools_scm'."
        )
        print(error_info)
        raise ModuleNotFoundError(str(e) + ". " + error_info)
    else:
        from pathlib import Path

        path = Path(__file__).resolve()
        __version__ = get_version(root="..", relative_to=path)
