import argparse
from importlib.metadata import PackageNotFoundError, version

from .bout_v5_factory_upgrader import add_parser as add_factory_parser
from .bout_v5_format_upgrader import add_parser as add_format_parser
from .bout_v5_header_upgrader import add_parser as add_header_parser
from .bout_v5_input_file_upgrader import add_parser as add_input_parser
from .bout_v5_macro_upgrader import add_parser as add_macro_parser
from .bout_v5_physics_model_upgrader import add_parser as add_model_parser
from .bout_v5_xzinterpolation_upgrader import add_parser as add_xzinterp_parser

try:
    # This gives the version if the boututils package was installed
    __version__ = version("boutdata")
except PackageNotFoundError:
    # This branch handles the case when boututils is used from the git repo
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except (ModuleNotFoundError, LookupError):
        __version__ = "dev"


def main():
    # Parent parser that has arguments common to all subcommands
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--quiet", "-q", action="store_true", help="Don't print patches"
    )
    force_or_patch_group = common_args.add_mutually_exclusive_group()
    force_or_patch_group.add_argument(
        "--force", "-f", action="store_true", help="Make changes without asking"
    )
    force_or_patch_group.add_argument(
        "--patch-only", "-p", action="store_true", help="Print the patches and exit"
    )

    # Parent parser for commands that always take a list of files
    files_args = argparse.ArgumentParser(add_help=False)
    files_args.add_argument("files", action="store", nargs="+", help="Input files")

    parser = argparse.ArgumentParser(
        description="Upgrade BOUT++ source and input files to newer versions"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subcommand = parser.add_subparsers(title="subcommands", required=True)
    v5_subcommand = subcommand.add_parser(
        "v5", help="BOUT++ v5 upgrades"
    ).add_subparsers(title="v5 subcommands", required=True)

    add_factory_parser(v5_subcommand, common_args, files_args)
    add_format_parser(v5_subcommand, common_args, files_args)
    add_header_parser(v5_subcommand, common_args)
    add_input_parser(v5_subcommand, common_args, files_args)
    add_macro_parser(v5_subcommand, common_args, files_args)
    add_model_parser(v5_subcommand, common_args, files_args)
    add_xzinterp_parser(v5_subcommand, common_args, files_args)

    args = parser.parse_args()
    args.func(args)
