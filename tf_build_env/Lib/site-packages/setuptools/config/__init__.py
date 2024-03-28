"""For backward compatibility, expose main functions from
``setuptools.config.setupcfg``
"""
import warnings
from functools import wraps
from textwrap import dedent
from typing import Callable, TypeVar, cast

from .._deprecation_warning import SetuptoolsDeprecationWarning
from . import setupcfg

Fn = TypeVar("Fn", bound=Callable)

__all__ = ('parse_configuration', 'read_configuration')


def _deprecation_notice(fn: Fn) -> Fn:
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        msg = f"""\
        As setuptools moves its configuration towards `pyproject.toml`,
        `{__name__}.{fn.__name__}` became deprecated.

        For the time being, you can use the `{setupcfg.__name__}` module
        to access a backward compatible API, but this module is provisional
        and might be removed in the future.
        """
        warnings.warn(dedent(msg), SetuptoolsDeprecationWarning, stacklevel=2)
        return fn(*args, **kwargs)

    return cast(Fn, _wrapper)


read_configuration = _deprecation_notice(setupcfg.read_configuration)
parse_configuration = _deprecation_notice(setupcfg.parse_configuration)
