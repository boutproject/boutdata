from itertools import chain
from typing import TypedDict
from warnings import warn

from boutdata.data import BoutOptionsFile

from .bout_v5_input_file_upgrader import add_parser_general, run_general


class Replacement(TypedDict):
    old: str
    new: str


REPLACEMENTS = []
DELETED = []

NEW_NAMES: dict[str, str | dict[str, list[str]]] = {
    "collisions": {
        "braginskii_collisions": [
            "electron_electron",
            "electron_ion",
            "electron_neutral",
            "ion_ion",
            "ion_neutral",
            "neutral_neutral",
            "ei_multiplier",
            "diagnose",
        ],
        "braginskii_friction": ["frictional_heating", "diagnose"],
        "braginskii_heat_exchange": ["diagnose"],
    },
    "electron_viscosity": "braginskii_electron_viscosity",
    "ion_viscosity": "braginskii_ion_viscosity",
    "thermal_force": "braginskii_thermal_force",
}

# If name in `components` and type not given then change the name in components and heading; if a multi-replacement, duplicate the section
# If type is same as name, duplicate section
# If name in type of any component, change it there (don't duplicate)


def split_list_string(list_string: str) -> tuple[list[str], bool]:
    result = [tname.strip() for tname in list_string.split(",")]
    open_paren = result[0][0] == "("
    close_paren = result[-1][-1] == ")"
    if open_paren != close_paren:
        warn(f'Unmatched parentheses around "{list_string}"')
    if open_paren:
        result[0] = result[0][1:]
    if close_paren:
        result[-1] = result[-1][:-1]
    return result, open_paren and close_paren


def rename_simple_component(
    options_file: BoutOptionsFile, section_name: str
) -> list[str]:
    """Rename a component when its type is the same as its name."""
    if section_name in NEW_NAMES:
        new_names = NEW_NAMES[section_name]
        if isinstance(new_names, dict):
            old_section = options_file.getSection(section_name)
            new_components = []
            for new_name, configs in new_names.items():
                new_components.append(new_name)
                new_section = options_file.getSection(new_name)
                for conf in configs:
                    if conf in old_section:
                        new_section[conf] = old_section[conf]
            return new_components
        else:
            options_file.rename(section_name, new_names)
            return [new_names]
    return [section_name]


def update_component_names(options_file: BoutOptionsFile) -> None:
    """Change the names of closure-related components to reflect the refactor"""
    has_collisions = False
    recycling_component = ""
    old_components, has_parens = split_list_string(options_file["hermes:components"])
    new_components = []
    for section_name in old_components:
        section = options_file.getSection(section_name)
        # Component type is set explicitly
        if "type" in section:
            old_types, types_have_parens = split_list_string(section["type"])
            # If component name and type match, treat similarly to if no type were given (see below)
            if len(old_types) == 1 and old_types[0] == section_name:
                has_collisions = has_collisions or section_name == "collisions"
                new_types = rename_simple_component(options_file, section_name)
                for t in new_types:
                    options_file[f"{t}:type"] = (
                        ("(" if types_have_parens else "")
                        + t
                        + (")" if types_have_parens else "")
                    )
                new_components.extend(new_types)
            # Otherwise simply replace any type-names that need changing
            else:
                has_collisions = has_collisions or section_name in old_types
                new_types = list(
                    chain.from_iterable(
                        [nt] if isinstance((nt := NEW_NAMES.get(t, t)), str) else nt
                        for t in old_types
                    )
                )
                if new_types != old_types:
                    section["type"] = (
                        ("(" if types_have_parens else "")
                        + ", ".join(new_types)
                        + (")" if types_have_parens else "")
                    )
                new_components.append(section_name)
        # Component type is same as component name
        else:
            has_collisions = has_collisions or section_name == "collisions"
            new_types = rename_simple_component(options_file, section_name)
            new_components.extend(new_types)
        if "recycling" in new_types:
            recycling_component = section_name
    # Add braginskii_conduction to the end of the list of components
    if has_collisions:
        new_components.append("braginskii_conduction")
        # Make sure recycling is evaluated after conduction
        if recycling_component != "":
            new_components.remove(recycling_component)
            new_components.append(recycling_component)
    if new_components != old_components:
        options_file["hermes:components"] = (
            ("(" if has_parens else "")
            + ", ".join(new_components)
            + (")" if has_parens else "")
        )


def run(args) -> None:
    run_general(
        REPLACEMENTS, DELETED, args, additional_modifications=update_component_names
    )


def add_parser(subcommand, default_args, files_args):
    return add_parser_general(subcommand, default_args, files_args, run, "Hermes-3")
