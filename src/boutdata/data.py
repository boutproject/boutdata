"""Provides a class BoutData which makes access to code inputs and
outputs easier. Creates a tree of maps, inspired by approach used in
OMFIT

"""

import copy
import glob
import io
import os
import re
from collections import OrderedDict, UserDict
from multiprocessing import Pipe, Process, RawArray

import numpy

# These are imported to be used by 'eval' in
# BoutOptions.evaluate_scalar() and BoutOptionsFile.evaluate().
# Change the names to match those used by C++/BOUT++
from numpy import abs as abs  # noqa: F401
from numpy import arccos as acos  # noqa: F401
from numpy import arccosh as acosh  # noqa: F401
from numpy import arcsin as asin  # noqa: F401
from numpy import arcsinh as asinh  # noqa: F401
from numpy import arctan as atan  # noqa: F401
from numpy import arctan2 as atan2  # noqa: F401
from numpy import arctanh as atanh  # noqa: F401
from numpy import (  # noqa: F401  # noqa: F401
    ceil,
    cos,
    cosh,
    exp,
    floor,
    log,
    log10,
    pi,
    round,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from numpy import power as pow  # noqa: F401

from boutdata.collect import (
    _check_fieldperp_attributes,
    _collect_from_one_proc,
    _get_grid_info,
    collect,
    create_cache,
    findVar,
)
from boututils.boutarray import BoutArray
from boututils.boutwarnings import alwayswarn
from boututils.datafile import DataFile
from boututils.run_wrapper import determineNumberOfCPUs


class CaseInsensitiveDict(UserDict):
    def __missing__(self, key):
        return CaseInsensitiveDict({key: CaseInsensitiveDict()})

    def __getitem__(self, key):
        return self.data[key.lower()][1]

    def __setitem__(self, key, value):
        self.data[key.lower()] = (key, value)

    def __delitem__(self, key):
        del self.data[key.lower()]

    def __iter__(self):
        return (key for key, _ in self.data.values())

    def __contains__(self, key):
        return key.lower() in self.data

    def __repr__(self):
        return repr({key: value for key, value in self.data.values()})


class BoutOptions(object):
    """This class represents a tree structure. Each node (BoutOptions
    object) can have several sub-nodes (sections), and several
    key-value pairs.

    Parameters
    ----------
    name : str, optional
        Name of the root section (default: "root")
    parent : BoutOptions, optional
        A parent BoutOptions object (default: None)

    Examples
    --------

    >>> optRoot = BoutOptions()  # Create a root

    Specify value of a key in a section "test"
    If the section does not exist then it is created

    >>> optRoot.getSection("test")["key"] = 4

    Get the value of a key in a section "test"
    If the section does not exist then a KeyError is raised

    >>> print(optRoot["test"]["key"])
    4

    To pretty print the options

    >>> print(optRoot)
    [test]
    key = 4

    """

    def __init__(self, name="root", parent=None):
        self._sections = CaseInsensitiveDict()
        self._keys = CaseInsensitiveDict()
        self._name = name
        self._parent = parent
        self.comments = CaseInsensitiveDict()
        self.inline_comments = CaseInsensitiveDict()
        self._comment_whitespace = CaseInsensitiveDict()

    def getSection(self, name):
        """Return a section object. If the section does not exist then it is
        created

        Parameters
        ----------
        name : str
            Name of the section to get/create

        Returns
        -------
        BoutOptions
            A new section with the original object as the parent

        """

        if name in self._sections:
            return self._sections[name]
        else:
            newsection = BoutOptions(name=name, parent=self)
            self._sections[name] = newsection
            return newsection

    def __getitem__(self, key):
        """
        First check if it's a section, then a value
        """

        key_parts = key.split(":", maxsplit=1)

        if len(key_parts) > 1:
            section = self[key_parts[0]]
            return section[key_parts[1]]

        if key in self._sections:
            return self._sections[key]

        if key not in self._keys:
            raise KeyError("Key '%s' not in section '%s'" % (key, self.path()))
        return self._keys[key]

    def __setitem__(self, key, value):
        """
        Set a key
        """
        if len(key) == 0:
            return

        key_parts = key.split(":", maxsplit=1)

        if len(key_parts) > 1:
            try:
                section = self[key_parts[0]]
            except KeyError:
                section = self.getSection(key_parts[0])
            section[key_parts[1]] = value
        else:
            self._keys[key] = value

    def __delitem__(self, key):
        key_parts = key.split(":", maxsplit=1)

        if len(key_parts) > 1:
            section = self[key_parts[0]]
            del section[key_parts[1]]
            return

        if key in self._sections:
            del self._sections[key]
        elif key in self._keys:
            del self._keys[key]
        else:
            raise KeyError(key)

    def __contains__(self, key):
        key_parts = key.split(":", maxsplit=1)

        if len(key_parts) > 1:
            if key_parts[0] in self:
                return key_parts[1] in self[key_parts[0]]
            return False

        return key in self._keys or key in self._sections

    __marker = object()

    def pop(self, key, default=__marker):
        """options.pop(k[,d]) -> v, remove specified key and return the
        corresponding value. If key is not found, d is returned if
        given, otherwise KeyError is raised.

        """
        return self._pop_impl(key, default)[0]

    def _pop_impl(self, key, default=__marker):
        """Private implementation of pop; also pops metadata"""
        key_parts = key.split(":", maxsplit=1)

        if len(key_parts) > 1:
            return self[key_parts[0]]._pop_impl(key_parts[1], default)

        if key in self._sections:
            value = self._sections.pop(key)
            name = self._name
            parent = self._parent
        elif key in self._keys:
            value = self._keys.pop(key)
            name = None
            parent = None
        elif default is self.__marker:
            raise KeyError(key)
        else:
            return default

        comment = self.comments.pop(key, None)
        inline_comment = self.inline_comments.pop(key, None)
        comment_whitespace = self._comment_whitespace.pop(key, None)

        return (value, name, parent, comment, inline_comment, comment_whitespace)

    def rename(self, old_name, new_name):
        """Rename old_name to new_name"""

        def setattr_nested(parent, key, attr, value):
            """Set one of the comment types on some nested section. Slightly
            complicated because the comment attributes are dicts, but
            we need to get the (possibly) nested parent section

            """
            # Don't set comment if it's None
            if value is None:
                return

            key_parts = key.split(":", maxsplit=1)
            if len(key_parts) > 1:
                setattr_nested(parent[key_parts[0]], key_parts[1], attr, value)
            else:
                getattr(parent, attr)[key] = value

        def check_is_section(parent, path):
            if path in parent and not isinstance(parent[path], BoutOptions):
                raise TypeError(
                    "'{}:{}' already exists and is not a section!".format(
                        parent._name, path
                    )
                )

        def ensure_sections(parent, path):
            """Make sure all the components of path in parent are sections"""
            path_parts = path.split(":", maxsplit=1)

            if len(path_parts) > 1:
                new_parent_name, child_name = path_parts
                check_is_section(parent, new_parent_name)
                parent.getSection(new_parent_name)
                ensure_sections(parent[new_parent_name], child_name)
            else:
                check_is_section(parent, path)
                parent.getSection(path)

        def rename_key(thing, new_name, old_name):
            """Rename a key in a dict while trying to preserve order, useful for
            minimising diffs"""
            return {new_name if k == old_name else k: v for k, v in thing.items()}

        def get_immediate_parent_and_child(path):
            """Get the immediate parent of path"""
            parent, _, child = path.rpartition(":")
            if parent and parent in self:
                return self[parent], child
            return self, path

        value = self[old_name]

        if isinstance(value, BoutOptions):
            # We're moving a section: make sure we don't clobber existing values
            ensure_sections(self, new_name)

            new_parent, new_child = get_immediate_parent_and_child(new_name)
            old_parent, old_child = get_immediate_parent_and_child(old_name)

            # Did we just add a new section?
            new_section = len(new_parent[new_child].keys()) == 0
            # Was it just a change in case?
            case_change = new_child.lower() == old_child.lower()

            # Renaming a child section just within the same parent section, we can
            # preserve the order
            if (new_parent is old_parent) and (new_section or case_change):
                # We just put a new section in, but it will have been
                # added at the end -- remove it so we can actually put
                # the new section in the same order as the original
                if new_section:
                    new_parent.pop(new_child)
                new_parent._sections = rename_key(
                    new_parent._sections, new_child, old_child
                )
                new_parent.comments = rename_key(
                    new_parent.comments, new_child, old_child
                )
                new_parent.inline_comments = rename_key(
                    new_parent.inline_comments, new_child, old_child
                )
                new_parent._comment_whitespace = rename_key(
                    new_parent._comment_whitespace, new_child, old_child
                )
                return

            # Now we're definitely moving into an existing section, so
            # update values and comments
            for key in value:
                self[new_name][key] = value[key]
                setattr_nested(self[new_name], key, "comments", value.comments.get(key))
                setattr_nested(
                    self[new_name],
                    key,
                    "inline_comments",
                    value.inline_comments.get(key),
                )
                setattr_nested(
                    self[new_name],
                    key,
                    "_comment_whitespace",
                    value._comment_whitespace.get(key),
                )
            _, _, _, comment, inline_comment, comment_whitespace = self._pop_impl(
                old_name
            )
        else:
            new_parent, new_child = get_immediate_parent_and_child(new_name)
            old_parent, old_child = get_immediate_parent_and_child(old_name)

            # Renaming a child key just within the same parent section, we can preserve
            # the order
            if new_parent is old_parent and ":" not in new_name:
                new_parent._keys = rename_key(new_parent._keys, new_child, old_child)
                new_parent.comments = rename_key(
                    new_parent.comments, new_child, old_child
                )
                new_parent.inline_comments = rename_key(
                    new_parent.inline_comments, new_child, old_child
                )
                new_parent._comment_whitespace = rename_key(
                    new_parent._comment_whitespace, new_child, old_child
                )
                return

            _, _, _, comment, inline_comment, comment_whitespace = self._pop_impl(
                old_name
            )
            self[new_name] = value

        # Update comments on new parent section
        setattr_nested(self, new_name, "comments", comment)
        setattr_nested(self, new_name, "inline_comments", inline_comment)
        setattr_nested(self, new_name, "_comment_whitespace", comment_whitespace)

    def path(self):
        """Returns the path of this section, joining together names of
        parents

        """

        if self._parent:
            return self._parent.path() + ":" + self._name
        return self._name

    def keys(self):
        """Returns all keys, including sections and values"""
        return list(self._sections) + list(self._keys)

    def sections(self):
        """Return a list of sub-sections"""
        return self._sections.keys()

    def values(self):
        """Return a list of values"""
        return self._keys.keys()

    def as_dict(self):
        """Return a nested dictionary of all the options."""
        dicttree = {name: self[name] for name in self.values()}
        dicttree.update({name: self[name].as_dict() for name in self.sections()})
        return dicttree

    def __len__(self):
        return len(self._sections) + len(self._keys)

    def __eq__(self, other):
        """Test if this BoutOptions is the same as another one."""
        if not isinstance(other, BoutOptions):
            return False
        if self is other:
            # other is a reference to the same object
            return True
        if len(self._sections) != len(other._sections):
            return False
        if len(self._keys) != len(other._keys):
            return False
        for secname, section in self._sections.items():
            if secname not in other or section != other[secname]:
                return False
        for key, value in self._keys.items():
            if key not in other or value != other[key]:
                return False
        return True

    def __iter__(self):
        """Iterates over all keys. First values, then sections"""
        for k in self._keys:
            yield k
        for s in self._sections:
            yield s

    def as_tree(self, indent=""):
        """Return a string formatted as a pretty version of the options tree"""
        text = self._name + "\n"

        for k in self._keys:
            text += indent + " |- " + k + " = " + str(self._keys[k]) + "\n"

        for s in self._sections:
            text += indent + " |- " + self._sections[s].as_tree(indent + " |  ")
        return text

    def __str__(self, basename=None, opts=None, f=None):
        if f is None:
            f = io.StringIO()
        if opts is None:
            opts = self

        def format_inline_comment(name, options):
            if name in options.inline_comments:
                f.write(
                    "{}{}".format(
                        options._comment_whitespace[name], options.inline_comments[name]
                    )
                )

        for key, value in opts._keys.items():
            if key in opts.comments:
                f.write("\n".join(opts.comments[key]) + "\n")
            f.write("{} = {}".format(key, value))
            format_inline_comment(key, opts)
            f.write("\n")

        for section in opts._sections.keys():
            section_name = basename + ":" + section if basename else section
            if section in opts.comments:
                f.write("\n".join(opts.comments[section]))
            if opts[section]._keys:
                f.write("\n[{}]".format(section_name))
                format_inline_comment(section, opts)
                f.write("\n")
            self.__str__(section_name, opts[section], f)

        return f.getvalue()

    def get_bool(self, name, default=None):
        """
        Convert an option value to a bool, in (almost) the same way as BOUT++.

        Warnings
        --------
        BOUT++ will convert any option value beginning with "y", "Y", "t", "T" or "1" to
        True, and any beginning with "n", "N", "f", "F" or "0" to False. Because
        BoutOptions converts option values to int and float, this method cannot be quite
        so permissive, and will raise an exception for ints other than 0 and 1 and for
        floats, which BOUT++ might convert to a bool.

        Parameters
        ----------
        name : str
            The name of the option to read
        default : bool, optional
            Value to return if the option is not present. If default is not provided an
            exception will be raised if the option is not present.
        """
        if default is not None and not isinstance(default, bool):
            raise ValueError(f'default "{default}" is not a bool')

        try:
            value = self[name]
        except KeyError:
            if default is None:
                raise
            else:
                return default

        if value == 1 or (
            isinstance(value, str) and value.lower() in ("y", "yes", "t", "true")
        ):
            return True
        elif value == 0 or (
            isinstance(value, str) and value.lower() in ("n", "no", "f", "false")
        ):
            return False

        raise ValueError(f"Could not convert {name}={value} to a bool")

    def evaluate_scalar(self, name):
        """
        Evaluate (recursively) scalar expressions
        """
        expression = self._substitute_expressions(name)

        # replace ^ with ** so that Python evaluates exponentiation
        expression = expression.replace("^", "**")

        return eval(expression)

    def _substitute_expressions(self, name):
        expression = str(self[name]).lower()
        expression = self._evaluate_section(expression, "")
        parent = self._parent
        while parent is not None:
            sectionname = parent._name
            if sectionname == "root":
                sectionname = ""
            expression = parent._evaluate_section(expression, sectionname)
            parent = parent._parent

        return expression

    def _evaluate_section(self, expression, nested_sectionname):
        # pass a nested section name so that we can traverse the options tree
        # rooted at our own level and each level above us so that we can use
        # relatively qualified variable names, e.g. if we are in section
        # 'foo:bar:baz' then a variable 'x' from section 'bar' could be called
        # 'bar:x' (found traversing the tree starting from 'bar') or
        # 'foo:bar:x' (found when traversing tree starting from 'foo').
        for var in self.values():
            if nested_sectionname != "":
                nested_name = nested_sectionname + ":" + var
            else:
                nested_name = var
            if re.search(
                r"(?<!:)\b" + re.escape(nested_name.lower()) + r"\b", expression.lower()
            ):
                # match nested_name only if not preceded by colon (which indicates more
                # nesting)
                expression = re.sub(
                    r"(?<!:)\b" + re.escape(nested_name.lower()) + r"\b",
                    "(" + self._substitute_expressions(var) + ")",
                    expression,
                )

        for subsection in self.sections():
            if nested_sectionname != "":
                nested_name = nested_sectionname + ":" + subsection
            else:
                nested_name = subsection
            expression = self.getSection(subsection)._evaluate_section(
                expression, nested_name
            )

        return expression


class BoutOptionsFile(BoutOptions):
    """Parses a BOUT.inp configuration file, producing a tree of
    BoutOptions.

    Slight differences from ConfigParser, including allowing values
    before the first section header.

    Parameters
    ----------
    filename : str, optional
        Path to file to read
    name : str, optional
        Name of root section (default: "root")
    gridfilename : str, optional
        If present, path to gridfile from which to read grid sizes (nx, ny, nz)
    nx, ny : int, optional
        - Specify sizes of grid, used when evaluating option strings
        - Cannot be given if gridfilename is specified
        - Must both be given if either is
        - If neither gridfilename nor nx, ny are given then will try to
          find nx, ny from (in order) the 'mesh' section of options,
          outputfiles in the same directory is the input file, or the grid
          file specified in the options file (used as a path relative to
          the current directory)
    nz : int, optional
        Use this value for nz when evaluating option expressions, if given.
        Overrides values found from input file, output files or grid files

    Examples
    --------

    >>> opts = BoutOptionsFile("BOUT.inp")
    >>> print(opts)   # Print all options in a tree
    root
    |- nout = 100
    |- timestep = 2
    ...

    >>> opts["All"]["scale"] # Value "scale" in section "All"
    1.0

    """

    # Characters that start a comment
    VALID_COMMENTS = ("#", ";")
    # Get not just the comment, but also the preceeding whitespace
    COMMENT_REGEX = re.compile(r"(.*?)(\s*)([{}].*)".format("".join(VALID_COMMENTS)))

    def __init__(
        self,
        filename="BOUT.inp",
        name="root",
        gridfilename=None,
        nx=None,
        ny=None,
        nz=None,
    ):
        BoutOptions.__init__(self, name)
        self.filename = filename
        self.gridfilename = gridfilename
        # Open the file
        with open(filename, "r") as f:
            # Go through each line in the file
            section = self  # Start with root section
            comments = []

            nr_line_iter = enumerate(f.readlines())
            for linenr, line in nr_line_iter:
                # First remove comments, either # or ;
                if line.lstrip().startswith(self.VALID_COMMENTS):
                    comments.append("#" + line.strip()[1:])
                    continue
                if line.strip() == "":
                    comments.append(line.strip())
                    continue

                comment_match = self.COMMENT_REGEX.search(line)
                if comment_match is not None:
                    line, comment_whitespace, inline_comment = comment_match.groups()
                    inline_comment = "#" + inline_comment.strip()[1:]
                else:
                    inline_comment = None
                    comment_whitespace = None

                # Check section headers
                startpos = line.find("[")
                endpos = line.find("]")
                if startpos != -1:
                    # A section heading
                    if endpos == -1:
                        raise SyntaxError("Missing ']' on line %d" % (linenr,))
                    line = line[(startpos + 1) : endpos].strip()

                    parent_section = self
                    while True:
                        scorepos = line.find(":")
                        if scorepos == -1:
                            sectionname = line
                            break
                        sectionname = line[0:scorepos]
                        line = line[(scorepos + 1) :]
                        parent_section = parent_section.getSection(sectionname)
                    section = parent_section.getSection(line)
                    if comments:
                        parent_section.comments[sectionname] = copy.deepcopy(comments)
                        comments = []
                    if inline_comment is not None:
                        parent_section.inline_comments[sectionname] = inline_comment
                        parent_section._comment_whitespace[sectionname] = (
                            comment_whitespace
                        )
                else:
                    # A key=value pair

                    eqpos = line.find("=")
                    if eqpos == -1:
                        # No '=', so just set to true
                        section[line.strip()] = True
                        value_name = line.strip()
                    else:
                        value = line[(eqpos + 1) :].strip()

                        # If the line contains unbalanced parentheses of brackets
                        # then continue reading
                        def count_brackets(s):
                            "Count net number of opening and closing brackets"
                            return (
                                s.count("(")
                                - s.count(")")
                                + s.count("[")
                                - s.count("]")
                            )

                        if count_brackets(value) != 0:
                            for cont_linenr, cont_line in nr_line_iter:
                                # Check for comments on continuing lines
                                comment_match = self.COMMENT_REGEX.search(cont_line)
                                if comment_match is not None:
                                    (
                                        cont_line,
                                        comment_whitespace,
                                        cont_inline_comment,
                                    ) = comment_match.groups()
                                    # Append inline comments
                                    if inline_comment is not None:
                                        inline_comment += " " + cont_inline_comment[1:]
                                    else:
                                        inline_comment = cont_inline_comment
                                value += " " + cont_line.strip()
                                if count_brackets(value) == 0:
                                    # Brackets now balanced
                                    break
                        try:
                            # Try to convert to an integer
                            value = int(value)
                        except ValueError:
                            try:
                                # Try to convert to float
                                value = float(value)
                            except ValueError:
                                # Leave as a string
                                pass

                        value_name = line[:eqpos].strip()
                        section[value_name] = value
                    if comments:
                        section.comments[value_name] = copy.deepcopy(comments)
                        comments = []
                    if inline_comment is not None:
                        section.inline_comments[value_name] = inline_comment
                        section._comment_whitespace[value_name] = comment_whitespace

        try:
            self.recalculate_xyz(nx=nx, ny=ny, nz=nz)
        except Exception as e:
            alwayswarn(
                "While building x, y, z coordinate arrays, an "
                "exception occured: "
                + str(e)
                + "\nEvaluating non-scalar options not available"
            )

    def recalculate_xyz(self, *, nx=None, ny=None, nz=None):
        """
        Recalculate the x, y avd z arrays used to evaluate expressions
        """
        # define arrays of x, y, z to be used for substitutions
        gridfile = None
        nzfromfile = None
        if self.gridfilename:
            if nx is not None or ny is not None:
                raise ValueError(
                    "nx or ny given as inputs even though "
                    "gridfilename was given explicitly, "
                    "don't know which parameters to choose"
                )
            with DataFile(self.gridfilename) as gridfile:
                self.nx = float(gridfile["nx"])
                self.ny = float(gridfile["ny"])
                try:
                    nzfromfile = gridfile["MZ"]
                except KeyError:
                    pass
        elif nx or ny:
            if nx is None:
                raise ValueError(
                    "nx not specified. If either nx or ny are given, then both must be."
                )
            if ny is None:
                raise ValueError(
                    "ny not specified. If either nx or ny are given, then both must be."
                )
            self.nx = nx
            self.ny = ny
        else:
            try:
                self.nx = self["mesh"].evaluate_scalar("nx")
                self.ny = self["mesh"].evaluate_scalar("ny")
            except KeyError:
                try:
                    # get nx, ny, nz from output files
                    from boutdata.collect import findFiles

                    file_list = findFiles(path=os.path.dirname("."), prefix="BOUT.dmp")
                    with DataFile(file_list[0]) as f:
                        self.nx = f["nx"]
                        self.ny = f["ny"]
                        nzfromfile = f["MZ"]
                except (IOError, KeyError):
                    try:
                        gridfilename = self["mesh"]["file"]
                    except KeyError:
                        gridfilename = self["grid"]
                    with DataFile(gridfilename) as gridfile:
                        self.nx = float(gridfile["nx"])
                        self.ny = float(gridfile["ny"])
                        try:
                            nzfromfile = float(gridfile["MZ"])
                        except KeyError:
                            pass
        if nz is not None:
            self.nz = nz
        else:
            try:
                self.nz = self["mesh"].evaluate_scalar("nz")
            except KeyError:
                try:
                    self.nz = self.evaluate_scalar("mz")
                except KeyError:
                    if nzfromfile is not None:
                        self.nz = nzfromfile
        mxg = self._keys.get("MXG", 2)
        myg = self._keys.get("MYG", 2)

        # make self.x, self.y, self.z three dimensional now so
        # that expressions broadcast together properly.
        self.x = numpy.linspace(
            (0.5 - mxg) / (self.nx - 2 * mxg),
            1.0 - (0.5 - mxg) / (self.nx - 2 * mxg),
            self.nx,
        )[:, numpy.newaxis, numpy.newaxis]
        self.y = (
            2.0
            * numpy.pi
            * numpy.linspace(
                (0.5 - myg) / self.ny,
                1.0 - (0.5 - myg) / self.ny,
                self.ny + 2 * myg,
            )[numpy.newaxis, :, numpy.newaxis]
        )
        self.z = (
            2.0
            * numpy.pi
            * numpy.linspace(0.5 / self.nz, 1.0 - 0.5 / self.nz, self.nz)[
                numpy.newaxis, numpy.newaxis, :
            ]
        )

        # Also create staggered versions of the coordinates
        self.xlow = numpy.linspace(
            -mxg / (self.nx - 2 * mxg),
            1.0 + (mxg - 1.0) / (self.nx - 2 * mxg),
            self.nx,
        )[:, numpy.newaxis, numpy.newaxis]
        self.ylow = (
            2.0
            * numpy.pi
            * numpy.linspace(
                -myg / self.ny,
                1.0 + (myg - 1.0) / self.ny,
                self.ny + 2 * myg,
            )[numpy.newaxis, :, numpy.newaxis]
        )
        self.zlow = (
            2.0
            * numpy.pi
            * numpy.linspace(0.0, 1.0 - 1.0 / self.nz, self.nz)[
                numpy.newaxis, numpy.newaxis, :
            ]
        )

    def evaluate(self, name, *, location="CELL_CENTRE"):
        """Evaluate (recursively) expressions

        Sections and subsections must be given as part of 'name',
        separated by colons

        Parameters
        ----------
        name : str
            Name of variable to evaluate, including sections and
            subsections

        """
        possible_locations = ["CELL_CENTRE", "CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"]
        if location not in possible_locations:
            raise ValueError(
                f"Unrecognised location {location}. Should be one of "
                f"{possible_locations}."
            )

        section = self
        split_name = name.split(":")
        for subsection in split_name[:-1]:
            section = section.getSection(subsection)
        expression = section._substitute_expressions(split_name[-1])

        # replace ^ with ** so that Python evaluates exponentiation
        expression = expression.replace("^", "**")

        # substitute for x, y and z coordinates
        for coord, coord_at_location in [
            ("x", "x" if location != "CELL_XLOW" else "xlow"),
            ("y", "y" if location != "CELL_YLOW" else "ylow"),
            ("z", "z" if location != "CELL_ZLOW" else "zlow"),
        ]:
            expression = re.sub(
                r"\b" + coord.lower() + r"\b", "self." + coord_at_location, expression
            )

        return eval(expression)

    def write(self, filename=None, overwrite=False):
        """Write to BOUT++ options file

        This method will throw an error rather than overwriting an existing
        file unless the overwrite argument is set to true.
        Note, no comments from the original input file are transferred to the
        new one.

        Parameters
        ----------
        filename : str
            Path of the file to write
            (defaults to path of the file that was read in)
        overwrite : bool
            If False then throw an exception if 'filename' already exists.
            Otherwise, just overwrite without asking.
            (default False)
        """
        if filename is None:
            filename = self.filename

        if not overwrite and os.path.exists(filename):
            raise ValueError(
                "Not overwriting existing file, cannot write output to " + filename
            )

        with open(filename, "w") as f:
            f.write(str(self))


class BoutOutputs(object):
    """Emulates a map class, represents the contents of a BOUT++ dmp
    files. Does not allow writing, only reading of data.  By default
    there is no cache, so each time a variable is read it is
    collected; if caching is set to True variables are stored once
    they are read.  Extra keyword arguments are passed through to
    collect.

    Parameters
    ----------
    path : str, optional
        Path to data files (default: ".")
    prefix : str, optional
        File prefix (default: "BOUT.dmp")
    suffix : str, optional
        File suffix (default: None, searches all file extensions)
    caching : bool, float, optional
        Switches on caching of data, so it is only read into memory
        when first accessed (default False) If caching is set to a
        number, it gives the maximum size of the cache in GB, after
        which entries will be discarded in first-in-first-out order to
        prevent the cache getting too big.  If the variable being
        returned is bigger than the maximum cache size, then the
        variable will be returned without being added to the cache,
        and the rest of the cache will be left (default: False)
    DataFileCaching : bool, optional
        Switch for creation of a cache of DataFile objects to be
        passed to collect so that DataFiles do not need to be
        re-opened to read each variable (default: True)
    info : bool, optional
        Print information about grid and data loading? (default: False)
    xguards : bool, optional
        Collect X boundary guard cells? (default: True)
        (Set to True to be consistent with the definition of nx)
    yguards : bool or "include_upper", optional
        Collect Y boundary guard cells? (default: False)
        If yguards=="include_upper" the y-boundary cells from the upper (second) target
        are also included.
    xind, yind, zind, tind : int, slice or list of int, optional
        Range of X, Y, Z or time indices to collect. Either a single
        index to collect, a list containing [start, end] (inclusive
        end), or a slice object (usual python indexing). Default is to
        fetch all indices
    parallel : bool or int, default False
        If set to True or 0, use the multiprocessing library to read data in parallel
        with the maximum number of available processors. If set to an int, use that many
        processes.
    tind_auto : bool, optional
        Read all files, to get the shortest length of time_indices. All data truncated
        to the shortest length.  Useful if writing got interrupted (default: False)

    Other parameters
    ----------------
    keyword arguments that are passed through to collect()

    Examples
    --------

    >>> d = BoutOutputs(".")  # Current directory
    >> d.keys()     # List all valid keys
    ['iteration',
     'zperiod',
     'MYSUB',
     ...
    ]

    >>> d.dimensions["ne"] # Get the dimensions of the field ne
    ('t', 'x', 'y', 'z')

    >>> d["ne"] # Read "ne" from data files
    BoutArray([[[[...]]]])

    >>> d = BoutOutputs(".", prefix="BOUT.dmp", caching=True) # Turn on caching

    """

    def __init__(
        self,
        path=".",
        prefix="BOUT.dmp",
        suffix=None,
        caching=False,
        DataFileCaching=True,
        info=False,
        xguards=True,
        yguards=False,
        tind=None,
        xind=None,
        yind=None,
        zind=None,
        tind_auto=False,
        parallel=False,
        **kwargs,
    ):
        """
        Initialise BoutOutputs object
        """
        self._path = path
        self._file0 = None
        # normalize prefix by removing trailing '.' if present
        self._prefix = prefix.removesuffix(".")
        self._caching = caching
        self._info = info
        self._xguards = xguards
        self._yguards = yguards
        self._kwargs = kwargs
        self._parallel = parallel
        if suffix is None:
            temp_file_list = glob.glob(os.path.join(self._path, self._prefix + "*"))
            latest_file = max(temp_file_list, key=os.path.getctime)
            self._suffix = latest_file.split(".")[-1]
        else:
            # normalize suffix by removing leading '.' if present
            self._suffix = suffix.removeprefix(".")
        if self._parallel is False:
            self._DataFileCaching = DataFileCaching
        else:
            # parallel functionality caches DataFiles in worker processes
            self._DataFileCaching = False

        if tind_auto:
            if self._kwargs.get("tind", None) is not None:
                raise ValueError("Cannot use 'tind' argument with 'tind_auto=True'")
            nt = len(
                self._collect(
                    "t_array",
                    path=self._path,
                    prefix=self._prefix,
                    tind_auto=True,
                    **self._kwargs,
                )
            )
            self._kwargs["tind"] = slice(nt)

        # Label for this data
        self.label = path

        self._file_list = glob.glob(
            os.path.join(path, self._prefix + "*" + self._suffix)
        )
        if suffix is not None:
            latest_file = max(self._file_list, key=os.path.getctime)
            # if suffix==None we already found latest_file

        # Check that the path contains some data
        if len(self._file_list) == 0:
            raise ValueError("ERROR: No data files found")

        with DataFile(latest_file) as f:
            self.grid_info, self.tind, self.xind, self.yind, self.zind = _get_grid_info(
                f,
                xguards=self._xguards,
                yguards=self._yguards,
                tind=tind,
                xind=xind,
                yind=yind,
                zind=zind,
                nfiles=len(self._file_list),
                all_vars_info=True,
            )

        if len(self._file_list) != self.grid_info["npes"]:
            alwayswarn("Too many data files, reading most recent ones")
        if (
            len(self._file_list) != self.grid_info["npes"]
            and self.grid_info["npes"] == 1
        ):
            # single output file
            # do like this to catch, e.g. either 'BOUT.dmp.nc' or 'BOUT.dmp.0.nc'
            self._file_list = [latest_file]
        else:
            # Re-create self._file_list so that it is sorted
            self._file_list = [
                os.path.join(path, self._prefix + "." + str(i) + "." + self._suffix)
                for i in range(self.grid_info["npes"])
            ]

        if self._DataFileCaching or self._parallel:
            # Keep reference to 0'th file, for reading attributes
            self._file0 = DataFile(self._file_list[0])

        if self._info:
            print(
                "mxsub = {} mysub = {} mz = {}\n".format(
                    self.grid_info["mxsub"],
                    self.grid_info["mysub"],
                    self.grid_info["nz"],
                )
            )
            print(
                "nxpe = {}, nype = {}, npes = {}\n".format(
                    self.grid_info["nxpe"],
                    self.grid_info["nype"],
                    self.grid_info["npes"],
                )
            )
            if self.grid_info["npes"] < len(self._file_list):
                print(
                    "WARNING: More files than expected ({})".format(
                        self.grid_info["npes"]
                    )
                )
            elif self.grid_info["npes"] > len(self._file_list):
                print(
                    "WARNING: Some files missing. Expected {}".format(
                        self.grid_info["npes"]
                    )
                )

        # Initialise private variables
        if self._caching:
            self._init_caching()
        if self._parallel is not False:
            self._init_parallel()
        self._DataFileCache = None

    def __del__(self):
        if self._parallel is not False:
            self._root_file.close()
            for worker, connection in self._workers:
                # Send None to terminate worker process cleanly
                connection.send(None)
                worker.join()
                connection.close()
        if self._file0 is not None:
            self._file0.close()

    def _init_caching(self):
        """
        Initialise private members used for caching of data variables
        """
        self._datacache = OrderedDict()
        if self._caching is not True:
            # Track the size of _datacache and limit it to a maximum of _caching
            try:
                # Check that _caching is a number of some sort
                float(self._caching)
            except ValueError:
                raise ValueError(
                    "BoutOutputs: Invalid value for caching argument. Caching should "
                    "be either a number (giving the maximum size of the cache in GB), "
                    "True for unlimited size or False for no caching."
                )
            self._datacachesize = 0
            self._datacachemaxsize = self._caching * 1.0e9

    def _init_parallel(self):
        """
        Initialise private members used for parallel reading
        """
        if self._parallel is True or self._parallel == 0:
            self._parallel = determineNumberOfCPUs()
        if not isinstance(self._parallel, int) or self._parallel <= 0:
            raise ValueError(
                "Passed or found inconsistent value %i for number of processes",
                self._parallel,
            )

        if self._parallel > self.grid_info["npes"]:
            # Using current self._parallel, some workers would have no work
            self._parallel = self.grid_info["npes"]

        # Open the 0'th file so we can read scalars without the worker processes
        self._root_file = DataFile(self._file_list[0])

        # Need to initialise all workers with a shared memory buffer to write to
        dim_sizes = tuple(self.grid_info["sizes"][d] for d in ("t", "x", "y", "z"))
        self._shared_buffer_raw = RawArray("d", int(numpy.prod(dim_sizes)))
        self._shared_buffer = numpy.reshape(
            numpy.frombuffer(self._shared_buffer_raw), dim_sizes
        )

        # Work out which files to assign to which workers
        min_files_per_proc = int(self.grid_info["npes"]) // self._parallel
        extra_files = int(self.grid_info["npes"]) % self._parallel
        files_per_proc = [min_files_per_proc] * (self._parallel - extra_files) + [
            min_files_per_proc + 1
        ] * extra_files
        # self._workers is a list of pairs of (worker, connection)
        self._workers = []
        filenum = 0
        for i in range(self._parallel):
            parent_connection, child_connection = Pipe()
            proc_list = tuple(p for p in range(filenum, filenum + files_per_proc[i]))
            filenum = filenum + files_per_proc[i]
            worker = Process(
                target=self._worker_function,
                args=(child_connection, proc_list, self._shared_buffer_raw),
            )
            worker.start()
            self._workers.append((worker, parent_connection))

    def keys(self):
        """Return a list of available variable names"""
        return self.grid_info["varNames"]

    @property
    def dimensions(self):
        """Accesss a dict of dimensions of the variables"""
        return self.grid_info["dimensions"]

    def evolvingVariables(self):
        """Return a list of names of time-evolving variables"""
        return self.grid_info["evolvingVariableNames"]

    def get_attribute(self, variable, attrname):
        """Get an attribute of a variable

        Parameters
        ----------
        variable : str
            Name of variable to get attribute from
        attrname : str
            Name of attribute

        Returns
        -------
        Value of attribute
        """
        if self._file0 is None:
            with DataFile(self._file_list[0]) as f:
                return f.attributes(variable)[attrname]
        else:
            return self._file0.attributes(variable)[attrname]

    def get_file_attribute(self, attrname):
        """Get an attribute of the output files.

        Attribute is taken from the rank-0 file. No checking is done that the attribute
        is consistent between all the output files.

        Parameters
        ----------
        attrname : str
            Name of attribute

        Returns
        -------
        Value of attribute
        """
        if self._file0 is None:
            with DataFile(self._file_list[0]) as f:
                return f.read_file_attribute(attrname)
        else:
            return self._file0.read_file_attribute(attrname)

    def list_file_attributes(self):
        """List all file attributes of output files

        List is taken from the rank-0 file. No checking is done that the file attributes
        are consistent between all the output files.

        Returns
        -------
        List of str
            Names of the file attributes
        """
        if self._file0 is None:
            with DataFile(self._file_list[0]) as f:
                return f.list_file_attributes()
        else:
            return self._file0.list_file_attributes()

    def redistribute(self, npes, nxpe=None, mxg=2, myg=2, include_restarts=True):
        """Create a new set of dump files for npes processors.

        Useful for restarting simulations using more or fewer processors.

        Existing data and restart files are kept in the directory
        "redistribution_backups". redistribute() will fail if this
        directory already exists, to avoid overwriting anything

        Parameters
        ----------
        npes : int
            Number of new files to create
        nxpe : int, optional
            If nxpe is None (the default), then an 'optimal' number will be
            selected automatically
        mxg, myg : int, optional
            Number of guard cells in x, y (default: 2)
        include_restarts : bool, optional
            If True, then restart.redistribute will be used to
            redistribute the restart files also (default: True)

        """
        from os import mkdir, path, rename

        from boutdata.processor_rearrange import (
            create_processor_layout,
            get_processor_layout,
        )

        # use get_processor_layout to get nx, ny
        old_processor_layout = get_processor_layout(
            DataFile(self._file_list[0]), has_t_dimension=True, mxg=mxg, myg=myg
        )
        nx = old_processor_layout.nx
        ny = old_processor_layout.ny
        mxg = old_processor_layout.mxg
        myg = old_processor_layout.myg

        # calculate new processor layout
        new_processor_layout = create_processor_layout(
            old_processor_layout, npes, nxpe=nxpe
        )
        nxpe = new_processor_layout.nxpe
        nype = new_processor_layout.nype
        mxsub = new_processor_layout.mxsub
        mysub = new_processor_layout.mysub

        # move existing files to backup directory
        # don't overwrite backup: os.mkdir will raise exception if directory already
        # exists
        backupdir = path.join(self._path, "redistribution_backups")
        mkdir(backupdir)
        for f in self._file_list:
            rename(f, path.join(backupdir, path.basename(f)))

        # create new output files
        outfile_list = []
        this_prefix = self._prefix
        if not this_prefix[-1] == ".":
            # ensure prefix ends with a '.'
            this_prefix = this_prefix + "."
        for i in range(npes):
            outpath = os.path.join(
                self._path, this_prefix + str(i) + "." + self._suffix
            )
            if self._suffix.split(".")[-1] in ["nc", "ncdf", "cdl"]:
                # set format option to DataFile explicitly to avoid creating netCDF3
                # files, which can only contain up to 2GB of data
                outfile_list.append(
                    DataFile(outpath, write=True, create=True, format="NETCDF4")
                )
            else:
                outfile_list.append(DataFile(outpath, write=True, create=True))

        # Create a DataFileCache, if needed
        if self._DataFileCaching:
            DataFileCache = create_cache(backupdir, self._prefix)
        else:
            DataFileCache = None
        # read and write the data
        for v in self.varNames:
            print("processing {}".format(v))
            data = collect(
                v,
                path=backupdir,
                prefix=self._prefix,
                xguards=True,
                yguards=True,
                info=False,
                datafile_cache=DataFileCache,
            )
            ndims = len(data.shape)

            # write data
            for i in range(npes):
                ix = i % nxpe
                iy = int(i / nxpe)
                outfile = outfile_list[i]
                if v == "NPES":
                    outfile.write(v, npes)
                elif v == "NXPE":
                    outfile.write(v, nxpe)
                elif v == "NYPE":
                    outfile.write(v, nype)
                elif v == "MXSUB":
                    outfile.write(v, mxsub)
                elif v == "MYSUB":
                    outfile.write(v, mysub)
                elif ndims == 0:
                    # scalar
                    outfile.write(v, data)
                elif ndims == 1:
                    # time evolving scalar
                    outfile.write(v, data)
                elif ndims == 2:
                    # Field2D
                    if data.shape != (nx + 2 * mxg, ny + 2 * myg):
                        # FieldPerp?
                        # check is not perfect, fails if ny=nz
                        raise ValueError(
                            "Error: Found FieldPerp '{}'. This case is not currently "
                            "handled by BoutOutputs.redistribute().".format(v)
                        )
                    outfile.write(
                        v,
                        data[
                            ix * mxsub : (ix + 1) * mxsub + 2 * mxg,
                            iy * mysub : (iy + 1) * mysub + 2 * myg,
                        ],
                    )
                elif ndims == 3:
                    # Field3D
                    if data.shape[:2] != (nx + 2 * mxg, ny + 2 * myg):
                        # evolving Field2D, but this case is not handled
                        # check is not perfect, fails if ny=nx and nx=nt
                        raise ValueError(
                            "Error: Found evolving Field2D '{}'. This case is not "
                            "currently handled by BoutOutputs.redistribute().".format(v)
                        )
                    outfile.write(
                        v,
                        data[
                            ix * mxsub : (ix + 1) * mxsub + 2 * mxg,
                            iy * mysub : (iy + 1) * mysub + 2 * myg,
                            :,
                        ],
                    )
                elif ndims == 4:
                    outfile.write(
                        v,
                        data[
                            :,
                            ix * mxsub : (ix + 1) * mxsub + 2 * mxg,
                            iy * mysub : (iy + 1) * mysub + 2 * myg,
                            :,
                        ],
                    )
                else:
                    print(
                        "ERROR: variable found with unexpected number of dimensions,",
                        ndims,
                    )

        for outfile in outfile_list:
            outfile.close()

        if include_restarts:
            print("processing restarts")
            from glob import glob

            from boutdata import restart

            restart_prefix = "BOUT.restart"
            restarts_list = glob(path.join(self._path, restart_prefix + "*"))

            # Move existing restart files to backup directory
            for f in restarts_list:
                rename(f, path.join(backupdir, path.basename(f)))

            # Redistribute restarts
            restart.redistribute(
                npes, path=backupdir, nxpe=nxpe, output=self._path, mxg=mxg, myg=myg
            )

    def _collect(self, varname):
        """Wrapper for collect to pass self._DataFileCache if necessary."""
        if self._parallel:
            return self._collect_parallel(varname)

        if self._DataFileCaching and self._DataFileCache is None:
            # Need to create the cache
            self._DataFileCache = create_cache(self._path, self._prefix)
        return collect(
            varname,
            datafile_cache=self._DataFileCache,
            path=self._path,
            prefix=self._prefix,
            info=self._info,
            xguards=self._xguards,
            yguards=self._yguards,
            tind=self.tind,
            xind=self.xind,
            yind=self.yind,
            zind=self.zind,
            **self._kwargs,
        )

    def _collect_parallel(self, varname):
        tind_auto = self._kwargs.get("tind_auto", False)
        strict = self._kwargs.get("strict", False)
        unsupported_kwargs = [
            k for k in self._kwargs if k not in ("tind_auto", "strict")
        ]
        if unsupported_kwargs:
            raise ValueError(
                "kwargs {} are not supported when parallel is not False".format(
                    unsupported_kwargs
                )
            )

        if tind_auto:
            raise ValueError("tind_auto not supported when parallel=True")

        if varname not in self.keys():
            if strict:
                raise ValueError("Variable '{}' not found".format(varname))
            else:
                varname = findVar(varname, self.keys())

        dimensions = self.grid_info["dimensions"][varname]
        var_attributes = self.grid_info["attributes"][varname]

        if not ("x" in dimensions or "y" in dimensions or "z" in dimensions):
            # No spatial dependence - read without using workers to preserve type
            if "t" in dimensions:
                return BoutArray(
                    self._root_file.read(
                        varname, ranges=[self.tind] + (len(dimensions) - 1) * [None]
                    ),
                    attributes=var_attributes,
                )
            else:
                # No time or space dimensions, so no slicing
                return BoutArray(
                    self._root_file.read(varname), attributes=var_attributes
                )
        elif any(dim not in ("t", "x", "y", "z") for dim in dimensions):
            raise ValueError(
                "Dimensions {} of {} contain spatial dimensions but also have "
                "dimensions that are not 't', 'x', 'y' or 'z'. This is not supported "
                "by parallel reading. Try reading with parallel=False".format(
                    dimensions, varname
                )
            )

        is_fieldperp = dimensions in (("t", "x", "z"), ("x", "z"))

        # Initialise buffer to zero
        self._shared_buffer[:] = 0.0

        for worker, connection in self._workers:
            connection.send((varname, is_fieldperp))

        yindex_global = None
        fieldperp_yproc = None

        for worker, connection in self._workers:
            temp_yindex, temp_fieldperp_yproc, temp_var_attributes = connection.recv()
            if is_fieldperp:
                (
                    yindex_global,
                    fieldperp_yproc,
                    var_attributes,
                ) = _check_fieldperp_attributes(
                    varname,
                    yindex_global,
                    temp_yindex,
                    temp_fieldperp_yproc,
                    fieldperp_yproc,
                    var_attributes,
                    temp_var_attributes,
                )

        global_slices = []
        if "t" in dimensions:
            global_slices.append(slice(None))
        else:
            global_slices.append(0)
        if "x" in dimensions:
            # Apply any requested step here, after collecting
            global_slices.append(slice(None, None, self.xind.step))
        else:
            global_slices.append(0)
        if "y" in dimensions:
            # Apply any requested step here, after collecting
            global_slices.append(slice(None, None, self.yind.step))
        else:
            global_slices.append(0)
        if "z" in dimensions:
            global_slices.append(slice(None))
        else:
            global_slices.append(0)
        global_slices = tuple(global_slices)

        return BoutArray(
            self._shared_buffer[global_slices].copy(), attributes=var_attributes
        )

    def __getstate__(self):
        new = self.__dict__.copy()
        new.pop("_file0")
        new.pop("_root_file")
        new.pop("_workers")
        return new

    def _worker_function(self, connection, proc_list, shared_buffer_raw):
        data_files = [DataFile(self._file_list[i]) for i in proc_list]
        dim_sizes = tuple(self.grid_info["sizes"][d] for d in ("t", "x", "y", "z"))
        shared_buffer = numpy.reshape(numpy.frombuffer(shared_buffer_raw), dim_sizes)
        while True:
            args = connection.recv()
            if args is None:
                # Terminate process cleanly
                for f in data_files:
                    f.close()
                connection.close()
                return 0

            varname, is_fieldperp = args

            yindex_global = None
            fieldperp_yproc = None
            var_attributes = None

            for i, f in zip(proc_list, data_files):
                temp_yindex, temp_var_attributes = _collect_from_one_proc(
                    i,
                    f,
                    varname,
                    result=shared_buffer,
                    is_fieldperp=is_fieldperp,
                    grid_info=self.grid_info,
                    dimensions=self.grid_info["dimensions"][varname],
                    tind=self.tind,
                    xind=self.xind,
                    yind=self.yind,
                    zind=self.zind,
                    xguards=self._xguards,
                    yguards=self._yguards,
                    info=self._info,
                    parallel_read=True,
                )
                if is_fieldperp:
                    (
                        yindex_global,
                        fieldperp_yproc,
                        var_attributes,
                    ) = _check_fieldperp_attributes(
                        varname,
                        yindex_global,
                        temp_yindex,
                        i // self.grid_info["nxpe"],
                        fieldperp_yproc,
                        var_attributes,
                        temp_var_attributes,
                    )

            connection.send((yindex_global, fieldperp_yproc, var_attributes))

    def __len__(self):
        return len(self.varNames)

    def __getitem__(self, name):
        """Reads a variable

        Caches result and returns later if called again, if caching is
        turned on for this instance

        """

        if self._caching:
            if name not in self._datacache.keys():
                item = self._collect(name)
                if self._caching is not True:
                    itemsize = item.nbytes
                    if itemsize > self._datacachemaxsize:
                        return item
                    self._datacache[name] = item
                    self._datacachesize += itemsize
                    while self._datacachesize > self._datacachemaxsize:
                        self._removeFirstFromCache()
                else:
                    self._datacache[name] = item
                return item
            else:
                return self._datacache[name]
        else:
            # Collect the data from the repository
            data = self._collect(name)
            return data

    def _removeFirstFromCache(self):
        """Pop the first item from the OrderedDict _datacache"""
        item = self._datacache.popitem(last=False)
        self._datacachesize -= item[1].nbytes

    def __iter__(self):
        """Iterate through all keys, starting with "options" then going
        through all variables for _caching_collect

        """
        for k in self.varNames:
            yield k

    def __str__(self, indent=""):
        """Print a pretty version of the tree"""
        text = ""
        for k in self.varNames:
            text += indent + k + "\n"

        return text


def BoutData(path=".", prefix="BOUT.dmp", caching=False, **kwargs):
    """Returns a dictionary, containing the contents of a BOUT++ output
    directory.

    Does not allow writing, only reading of data.  By default there is
    no cache, so each time a variable is read it is collected; if
    caching is set to True variables are stored once they are read.

    Parameters
    ----------
    path : str, optional
        Path to data files (default: ".")
    prefix : str, optional
        File prefix (default: "BOUT.dmp")
    caching : bool, float, optional
        Switches on caching of data, so it is only read into memory
        when first accessed (default False) If caching is set to a
        number, it gives the maximum size of the cache in GB, after
        which entries will be discarded in first-in-first-out order to
        prevent the cache getting too big.  If the variable being
        returned is bigger than the maximum cache size, then the
        variable will be returned without being added to the cache,
        and the rest of the cache will be left (default: False)
    DataFileCaching : bool, optional
        Switch for creation of a cache of DataFile objects to be
        passed to collect so that DataFiles do not need to be
        re-opened to read each variable (default: True)
    **kwargs
        Keyword arguments that are passed through to collect()

    Returns
    -------
    dict
        Contents of a BOUT++ output directory, including options and
        output files

    Examples
    --------

    >>> d = BoutData(".")  # Current directory

    >>> d.keys()     # List all valid keys

    >>> print(d["options"])  # Prints tree of options

    >>> d["options"]["nout"]   # Value of nout in BOUT.inp file

    >>> print(d["outputs"])    # Print available outputs

    >>> d["outputs"]["ne"] # Read "ne" from data files

    >>> d = BoutData(".", prefix="BOUT.dmp", caching=True) # Turn on caching

    """

    data = {}  # Map for the result

    data["path"] = path

    # Options from BOUT.inp file
    data["options"] = BoutOptionsFile(os.path.join(path, "BOUT.inp"), name="options")

    # Output from .dmp.* files
    data["outputs"] = BoutOutputs(path, prefix=prefix, caching=caching, **kwargs)

    return data
