import argparse
import copy
import re
import textwrap

from .common import apply_or_display_patch

# List of macros, their replacements and what header to find them
# in. Each element should be a dict with "old", "new" and "headers"
# keys, with "old" and "new" values being strings, and "headers" being a
# list of strings. "new" can also be None if the macro has been removed, which
# will cause an error to be printed if the macro is found.
MACRO_REPLACEMENTS = [
    {
        "old": "REVISION",
        "new": "bout::version::revision",
        "headers": ["bout/revision.hxx"],
        "macro": False,
        "always_defined": True,
    },
    {
        "old": "BOUT_VERSION_DOUBLE",
        "new": "bout::version::as_double",
        "headers": ["bout/version.hxx", "bout.hxx"],
        "macro": False,
        "always_defined": True,
    },
    {
        "old": "BOUT_VERSION_STRING",
        "new": "bout::version::full",
        "headers": ["bout/version.hxx", "bout.hxx"],
        "macro": False,
        "always_defined": True,
    },
    # Next one is not technically a macro, but near enough
    {
        "old": "BOUT_VERSION",
        "new": "bout::version::full",
        "headers": ["bout/version.hxx", "bout.hxx"],
        "macro": False,
        "always_defined": True,
    },
    {
        "old": "BACKTRACE",
        "new": "BOUT_USE_BACKTRACE",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_ARKODE",
        "new": "BOUT_HAS_ARKODE",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_CVODE",
        "new": "BOUT_HAS_CVODE",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_HDF5",
        "new": None,
        "headers": [],
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_IDA",
        "new": "BOUT_HAS_IDA",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_LAPACK",
        "new": "BOUT_HAS_LAPACK",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "LAPACK",
        "new": "BOUT_HAS_LAPACK",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_NETCDF",
        "new": "BOUT_HAS_NETCDF",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_PETSC",
        "new": "BOUT_HAS_PETSC",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_PRETTY_FUNCTION",
        "new": "BOUT_HAS_PRETTY_FUNCTION",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HAS_PVODE",
        "new": "BOUT_HAS_PVODE",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "TRACK",
        "new": "BOUT_USE_TRACK",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "NCDF4",
        "new": "BOUT_HAS_NETCDF",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "NCDF",
        "new": "BOUT_HAS_LEGACY_NETCDF",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "HDF5",
        "new": None,
        "headers": [],
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "DEBUG_ENABLED",
        "new": "BOUT_USE_OUTPUT_DEBUG",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "BOUT_FPE",
        "new": "BOUT_USE_SIGFPE",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "LOGCOLOR",
        "new": "BOUT_USE_COLOR",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
    {
        "old": "OPENMP_SCHEDULE",
        "new": "BOUT_OPENMP_SCHEDULE",
        "headers": "bout/build_config.hxx",
        "macro": True,
        "always_defined": True,
    },
]


def fix_include_version_header(old, headers, source):
    """Make sure version.hxx header is included"""

    if not isinstance(headers, list):
        headers = [headers]

    # If header is already included, we can skip this fix
    for header in headers:
        if (
            re.search(rf'^#\s*include.*(<|"){header}(>|")', source, flags=re.MULTILINE)
            is not None
        ):
            return source

    # If the old macro isn't in the file, we can skip this fix
    if re.search(rf"\b{old}\b", source) is None:
        return source

    # Now we want to find a suitable place to stick the new include
    # Good candidates are includes of BOUT++ headers
    includes = []
    source_lines = source.splitlines()
    for linenumber, line in enumerate(source_lines):
        if re.match(r"^#\s*include.*bout/", line):
            includes.append(linenumber)
        if re.match(r"^#\s*include.*physicsmodel", line):
            includes.append(linenumber)

    if includes:
        last_include = includes[-1] + 1
    else:
        # No suitable includes, so just stick at the top of the file
        last_include = 0
    source_lines.insert(last_include, f'#include "{headers[0]}"')

    return "\n".join(source_lines)


def fix_ifdefs(old, source):
    """Remove any code inside #ifdef/#ifndef blocks that would now not be compiled"""
    source_lines = source.splitlines()

    # Something to keep track of nested sections
    in_ifdef = None
    # List of (#ifdef or #ifndef, dict of start/else/end lines)
    macro_blocks = []
    for linenumber, line in enumerate(source_lines):
        if_def = re.match(r"#\s*(ifn?def)\s*(.*)", line)
        else_block = re.match(r"#\s*else", line)
        endif = re.match(r"#\s*endif", line)
        if not (if_def or else_block or endif):
            continue
        # Now we need to keep track of whether we're inside an
        # interesting #ifdef/ifndef, as they might be nested, and we
        # want to find the matching #endif and #else
        if endif:
            if in_ifdef is not None:
                in_ifdef -= 1
            if in_ifdef == 0:
                in_ifdef = None
                macro_blocks[-1]["end"] = linenumber
            continue
        if else_block:
            if in_ifdef == 1:
                macro_blocks[-1]["else"] = linenumber
            continue
        if if_def.group(2) == old:
            in_ifdef = 1
            macro_blocks.append({"start": linenumber, "if_def_type": if_def.group(1)})
        elif in_ifdef is not None:
            in_ifdef += 1

    if macro_blocks == []:
        return source

    # Get all of the lines to be removed
    lines_to_remove = set()
    for block in macro_blocks:
        lines_to_remove |= set(block.values())
        if block["if_def_type"] == "ifdef":
            if "else" in block:
                # Delete the #else block for #ifdef
                lines_to_remove |= set(range(block["else"], block["end"]))
        else:
            # Keep the #else block for #ifndef if there is one, otherwise remove the
            # whole block
            lines_to_remove |= set(
                range(block["start"], block.get("else", block["end"]))
            )

    # Apparently this is actually the best way of removing a bunch of (possibly)
    # non-contiguous indices
    modified_lines = [
        line for num, line in enumerate(source_lines) if num not in lines_to_remove
    ]

    return "\n".join(modified_lines)


def fix_always_defined_macros(old, new, source):
    """Fix '#ifdef's that should become plain '#if'"""
    new_source = re.sub(rf"#ifdef\s+{old}\b", rf"#if {new}", source)
    return re.sub(rf"#ifndef\s+{old}\b", rf"#if !{new}", new_source)


def fix_replacement(old, new, source):
    """Straight replacements"""
    return re.sub(rf'([^"_])\b{old}\b([^"_])', rf"\1{new}\2", source)


def apply_fixes(replacements, source):
    """Apply all fixes in this module"""
    modified = copy.deepcopy(source)

    for replacement in replacements:
        if replacement["new"] is None:
            print(
                f"{replacement['old']} has been removed, please delete from your code"
            )
            continue

        modified = fix_include_version_header(
            replacement["old"], replacement["headers"], modified
        )
        if replacement["macro"] and replacement["always_defined"]:
            modified = fix_always_defined_macros(
                replacement["old"], replacement["new"], modified
            )
        elif replacement["always_defined"]:
            modified = fix_ifdefs(replacement["old"], modified)
        modified = fix_replacement(replacement["old"], replacement["new"], modified)

    return modified


def add_parser(subcommand, default_args, files_args):
    parser = subcommand.add_parser(
        "macro",
        help="Fix macro defines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Fix macro defines for BOUT++ v4 -> v5

            Please note that this is only slightly better than dumb text replacement. It
            will fix the following:

            * replacement of macros with variables or new names
            * inclusion of correct headers for new variables
            * removal of #if(n)def/#endif blocks that do simple checks for the old
              macro, keeping the appriopriate part, if replaced by a variable
            * change '#if(n)def' for '#if (!)' if the replacment is always defined

            It will try not to replace quoted macro names, but may
            still replace them in strings or comments.

            Please check the diff output carefully!
            """
        ),
        parents=[default_args, files_args],
    )
    parser.set_defaults(func=run)


def run(args):
    for filename in args.files:
        with open(filename) as f:
            contents = f.read()
        original = copy.deepcopy(contents)

        modified = apply_fixes(MACRO_REPLACEMENTS, contents)
        apply_or_display_patch(
            filename, original, modified, args.patch_only, args.quiet, args.force
        )
