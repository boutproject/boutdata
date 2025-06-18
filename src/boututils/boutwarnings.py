"""
Wrappers for warnings functions.

Allows raising warnings that are always printed by default.
"""

import warnings


class AlwaysWarning(UserWarning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


warnings.simplefilter("always", AlwaysWarning)


def alwayswarn(message):
    warnings.warn(message, AlwaysWarning, stacklevel=2)


def defaultwarn(message):
    warnings.warn(message, stacklevel=2)
