from itertools import chain
from typing import TypedDict
import re

from .bout_v5_input_file_upgrader import add_parser_general, run_general
from boutdata.data import BoutOptionsFile

class Replacement(TypedDict):
    old: str
    new: str

REPLACEMENTS = []
DELETED = []

NEW_NAMES = {
    "collisions": ["braginskii_collisions", "braginskii_friction", "braginskii_heat_exchange"],
    "electron_viscosity": ["braginskii_electron_viscosity"],
    "ion_viscosity": ["braginskii_ion_viscosity"],
    "thermal_force": ["braginskii_thermal_force"]
}

COMPONENT_RE = re.compile(r"[+\-\w]+")


def update_component_names(options_file: BoutOptionsFile) -> None:
    """Change the names of closure-related components to reflect the refactor"""
    has_collisions = False
    recycling_component = ""
    components = re.findall(COMPONENT_RE, options_file["hermes:components"])
    for section_name in components:
        section = options_file.getSection(section_name)
        explicit_types = "type" in section
        types = [tname.strip() for tname in section["type"].split(",")] if explicit_types else [section_name]
        open_paren = types[0][0] == "("
        close_paren = types[-1][-1] == ")"
        if open_paren:
            types[0] = types[0][1:]
        if close_paren:
            types[-1] = types[-1][:-1]
        new_types = list(chain.from_iterable(NEW_NAMES.get(t, [t]) for t in types))
        has_collisions = has_collisions or "collisions" in types
        if "recycling" in new_types:
            recycling_component = section_name
        if new_types != types:
            section["type"] =  ("(" if open_paren else "") + ", ".join(new_types) + (")" if close_paren else "")
    # Add braginskii_conduction to the end of the list of components
    if has_collisions:
        components.append("braginskii_conduction")
        # Make sure recycling is evaluated after conduction
        if recycling_component != "":
            components.remove(recycling_component)
            components.append(recycling_component)
    options_file["hermes:components"] = "(" + ", ".join(components) + ")"

def run(args) -> None:
    run_general(REPLACEMENTS, DELETED, args, additional_modifications=update_component_names)
    
def add_parser(subcommand, default_args, files_args):
    return add_parser_general(subcommand, default_args, files_args, run, "Hermes-3")
