import setuptools.extern.jaraco.text as text

from pkg_resources import Requirement


def parse_strings(strs):
    """
    Yield requirement strings for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return text.join_continuation(map(text.drop_comment, text.yield_lines(strs)))


def parse(strs):
    """
    Deprecated drop-in replacement for pkg_resources.parse_requirements.
    """
    return map(Requirement, parse_strings(strs))
