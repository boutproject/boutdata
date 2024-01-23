""" Generic routines, useful for all data """

__all__ = []
__name__ = "boututils"

try:
    from importlib.metadata import PackageNotFoundError, version
except ModuleNotFoundError:
    from importlib_metadata import PackageNotFoundError, version
try:
    # This gives the version if the boututils package was installed
    __version__ = version(__name__)
except PackageNotFoundError:
    # This branch handles the case when boututils is used from the git repo
    try:
        from pathlib import Path

        from setuptools_scm import get_version

        path = Path(__file__).resolve()
        __version__ = get_version(root="..", relative_to=path)
    except (ModuleNotFoundError, LookupError):
        # ModuleNotFoundError if setuptools_scm is not installed.
        # LookupError if git is not installed, or the code is not in a git repo even
        # though it has not been installed.
        from warnings import warn

        warn(
            "'setuptools_scm' and git are required to get the version number when "
            "running boututils from the git repo. Please install 'setuptools_scm' and "
            "check 'git rev-parse HEAD' works. Setting __version__='dev' as a "
            "workaround."
        )
        __version__ = "dev"
