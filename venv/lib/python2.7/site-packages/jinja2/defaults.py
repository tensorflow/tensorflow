# -*- coding: utf-8 -*-
"""
    jinja2.defaults
    ~~~~~~~~~~~~~~~

    Jinja default filters and tags.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
from jinja2._compat import range_type
from jinja2.utils import generate_lorem_ipsum, Cycler, Joiner


# defaults for the parser / lexer
BLOCK_START_STRING = '{%'
BLOCK_END_STRING = '%}'
VARIABLE_START_STRING = '{{'
VARIABLE_END_STRING = '}}'
COMMENT_START_STRING = '{#'
COMMENT_END_STRING = '#}'
LINE_STATEMENT_PREFIX = None
LINE_COMMENT_PREFIX = None
TRIM_BLOCKS = False
LSTRIP_BLOCKS = False
NEWLINE_SEQUENCE = '\n'
KEEP_TRAILING_NEWLINE = False


# default filters, tests and namespace
from jinja2.filters import FILTERS as DEFAULT_FILTERS
from jinja2.tests import TESTS as DEFAULT_TESTS
DEFAULT_NAMESPACE = {
    'range':        range_type,
    'dict':         dict,
    'lipsum':       generate_lorem_ipsum,
    'cycler':       Cycler,
    'joiner':       Joiner
}


# default policies
DEFAULT_POLICIES = {
    'compiler.ascii_str':   True,
    'urlize.rel':           'noopener',
    'urlize.target':        None,
    'truncate.leeway':      5,
    'json.dumps_function':  None,
    'json.dumps_kwargs':    {'sort_keys': True},
}


# export all constants
__all__ = tuple(x for x in locals().keys() if x.isupper())
