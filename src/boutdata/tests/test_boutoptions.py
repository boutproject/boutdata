import textwrap

import pytest

from boutdata.data import BoutOptions


def test_getSection_nonexistent():
    options = BoutOptions()
    options.getSection("new")
    assert "new" in options


def test_get_set_item_value():
    options = BoutOptions()
    options["new"] = 5
    assert options["new"] == 5


def test_get_set_item_section():
    options = BoutOptions()
    options["section:new"] = 6
    assert "section" in options
    assert options["section"]["new"] == 6


def test_contains():
    options = BoutOptions()
    options["a:b:c"] = 42

    assert "a" in options
    assert "a:b" in options
    assert "a:b:c" in options
    assert "abc" not in options


def test_as_dict():
    options = BoutOptions()
    options["section:new"] = 7
    expected = {"section": {"new": 7}}
    assert options.as_dict() == expected


def test_rename_section_same_level():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section", "another")

    expected = {
        "top-level value": 0,
        "another": {"first": 1, "second": 2},
        "other top-level": 3,
    }
    assert "another" in options
    assert "section" not in options
    assert options.as_dict() == expected


def test_rename_value_same_level():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section:first", "section:third")

    expected = {
        "top-level value": 0,
        "section": {"third": 1, "second": 2},
        "other top-level": 3,
    }
    assert "section:third" in options
    assert "section:first" not in options
    assert options.as_dict() == expected


def test_rename_value_case_sensitive():
    options = BoutOptions()
    options["lower"] = 0

    options.rename("lower", "LOWER")

    expected = {"LOWER": 0}
    assert options.as_dict() == expected


def test_rename_section_case_sensitive():
    options = BoutOptions()
    options["lower:a"] = 0

    options.rename("lower", "LOWER")

    expected = {"LOWER": {"a": 0}}
    assert options.as_dict() == expected


def test_rename_section_deeper():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section", "another:layer")

    expected = {
        "top-level value": 0,
        "another": {
            "layer": {"first": 1, "second": 2},
        },
        "other top-level": 3,
    }
    assert "another" in options
    assert "section" not in options
    assert options.as_dict() == expected


def test_rename_section_into_other_section():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section1")
    section["first"] = 1
    section["second"] = 2
    section2 = options.getSection("section2")
    section2["third"] = 3
    section2["fourth"] = 4
    options["other top-level"] = 5

    options.rename("section1", "section2")

    expected = {
        "top-level value": 0,
        "section2": {"first": 1, "second": 2, "third": 3, "fourth": 4},
        "other top-level": 5,
    }
    assert options.as_dict() == expected
    assert "section2:third" in options
    assert "section1:first" not in options


def test_rename_value_deeper():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section:first", "section:subsection:first")

    expected = {
        "top-level value": 0,
        "section": {
            "second": 2,
            "subsection": {"first": 1},
        },
        "other top-level": 3,
    }
    assert "section:subsection:first" in options
    assert "section:first" not in options
    assert options.as_dict() == expected


def test_rename_value_into_other_section():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section1")
    section["first"] = 1
    section["second"] = 2
    section2 = options.getSection("section2")
    section2["third"] = 3
    section2["fourth"] = 4
    options["other top-level"] = 5

    options.rename("section1:first", "section2:first")

    expected = {
        "top-level value": 0,
        "section1": {"second": 2},
        "section2": {"first": 1, "third": 3, "fourth": 4},
        "other top-level": 5,
    }
    assert options.as_dict() == expected
    assert "section2:third" in options
    assert "section1:first" not in options


def test_path():
    options = BoutOptions("top level")
    options["a:b:c:d"] = 1
    section = options.getSection("a:b:c")

    assert section.path() == "top level:a:b:c"


def test_str():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    # lstrip to remove the first empty line
    expected = textwrap.dedent("""
        top-level value = 0
        other top-level = 3

        [section]
        first = 1
        second = 2
        """).lstrip()

    assert str(options) == expected


def test_get_bool():
    options = BoutOptions()

    for truelike in ["y", "Y", "yes", "Yes", "t", "T", "true", "True", 1]:
        options["truevalue"] = truelike
        assert options.get_bool("truevalue") is True
        assert options.get_bool("truevalue", True) is True
        assert options.get_bool("truevalue", False) is True
        with pytest.raises(ValueError):
            options.get_bool("truevalue", "not a bool")

    for falseelike in ["n", "N", "no", "No", "f", "F", "false", "False", 0]:
        options["falseevalue"] = falseelike
        assert options.get_bool("falseevalue") is False
        assert options.get_bool("falseevalue", True) is False
        assert options.get_bool("falseevalue", False) is False
        with pytest.raises(ValueError):
            options.get_bool("falsevalue", 1)

    with pytest.raises(KeyError):
        options.get_bool("missingoption")
    assert options.get_bool("missingoption", True) is True
    assert options.get_bool("missingoption", False) is False
    with pytest.raises(ValueError):
        options.get_bool("missingoption", "not a bool")

    for invalid in [
        "bar",
        "yaihets",
        "Yaxfus",
        "tueoxg",
        "Teouaig",
        "1uegxa",
        "naihets",
        "Naxfus",
        "fueoxg",
        "Feouaig",
        "0uegxa",
    ]:
        options["stringvalue"] = invalid
        with pytest.raises(ValueError):
            options.get_bool("stringvalue")
        with pytest.raises(ValueError):
            options.get_bool("stringvalue", True)
        with pytest.raises(ValueError):
            options.get_bool("stringvalue", False)
