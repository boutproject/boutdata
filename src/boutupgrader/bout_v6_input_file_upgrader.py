from .bout_v5_input_file_upgrader import add_parser_general, run_general

# This should be a list of dicts, each containing "old", "new" and optionally "new_values".
# The values of "old"/"new" keys should be the old/new names of input file values or
# sections. The value of "new_values" is a dict containing replacements for values of the
# option. "old_type" optionally specifies the type of the old value of the option; for
# example this is needed for special handling of boolean values.
REPLACEMENTS = [
    {"old": "timestep", "new": "solver:output_step"},
    {"old": "nout", "new": "solver:nout"},
    {"old": "grid", "new": "mesh:file"},
]

DELETED = []


def run(args):
    return run_general(REPLACEMENTS, DELETED, args)


def add_parser(subcommand, default_args, files_args):
    return add_parser_general(subcommand, default_args, files_args, run, "BOUT++ v6+")
