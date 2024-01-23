""" Routines for exchanging data to/from BOUT++ """

from boutdata.collect import collect, attributes
from boututils.boutarray import BoutArray
from boututils.boutwarnings import alwayswarn
from boututils.run_wrapper import (
    launch,
    launch_safe,
    shell,
    shell_safe,
    determineNumberOfCPUs,
    build_and_log,
)


__all__ = [
    "attributes",
    "collect",
    "gen_surface",
    "pol_slice",
    "BoutArray",
    "alwayswarn",
    "launch",
    "launch_safe",
    "shell",
    "shell_safe",
    "determineNumberOfCPUs",
    "build_and_log",
]

__name__ = "boutdata"

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
try:
    # This gives the version if the boutdata package was installed
    __version__ = version(__name__)
except PackageNotFoundError:
    # This branch handles the case when boutdata is used from the git repo
    try:
        from setuptools_scm import get_version
        from pathlib import Path

        path = Path(__file__).resolve()
        __version__ = get_version(root="..", relative_to=path)
    except (ModuleNotFoundError, LookupError):
        # ModuleNotFoundError if setuptools_scm is not installed.
        # LookupError if git is not installed, or the code is not in a git repo even
        # though it has not been installed.
        from warnings import warn

        warn(
            "'setuptools_scm' and git are required to get the version number when "
            "running boutdata from the git repo. Please install 'setuptools_scm' and "
            "check 'git rev-parse HEAD' works. Setting __version__='dev' as a "
            "workaround."
        )
        __version__ = "dev"
