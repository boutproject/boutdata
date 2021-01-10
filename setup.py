#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import setuptools

name = "boutdata"
root_path = Path(__file__).parent
init_path = root_path.joinpath(name, "__init__.py")
readme_path = root_path.joinpath("README.md")

with readme_path.open("r") as f:
    long_description = f.read()

setuptools.setup(
    name=name,
    author="Ben Dudson et al.",
    description="Python package for collecting BOUT++ data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boutproject/boutdata",
    project_urls={
        "Bug Tracker": "https://github.com/boutproject/boutdata/issues/",
        "Documentation": "https://bout-dev.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/boutproject/boutdata/",
    },
    packages=setuptools.find_packages(),
    keywords=[
        "bout++",
        "bout",
        "plasma",
        "physics",
        "data-extraction",
        "data-analysis",
        "data-visualization",
    ],
    use_scm_version=True,
    setup_requires=[
        "setuptools>=42",
        "setuptools_scm[toml]>=3.4",
        "setuptools_scm_git_archive",
    ],
    install_requires=[
        "sympy",
        "numpy",
        "matplotlib",
        "scipy",
        "boututils",
        "importlib-metadata ; python_version<'3.8'",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        (
            "License :: OSI Approved :: "
            "GNU Lesser General Public License v3 or later (LGPLv3+)"
        ),
        "Operating System :: OS Independent",
    ],
)
