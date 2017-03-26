# -*- coding: utf-8 -*-
"""
    markupsafe._native
    ~~~~~~~~~~~~~~~~~~

    Native Python implementation the C module is not compiled.

    :copyright: (c) 2010 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
from markupsafe import Markup
from markupsafe._compat import text_type


def escape(s):
    """Convert the characters &, <, >, ' and " in string s to HTML-safe
    sequences.  Use this if you need to display text that might contain
    such characters in HTML.  Marks return value as markup string.
    """
    if hasattr(s, '__html__'):
        return s.__html__()
    return Markup(text_type(s)
        .replace('&', '&amp;')
        .replace('>', '&gt;')
        .replace('<', '&lt;')
        .replace("'", '&#39;')
        .replace('"', '&#34;')
    )


def escape_silent(s):
    """Like :func:`escape` but converts `None` into an empty
    markup string.
    """
    if s is None:
        return Markup()
    return escape(s)


def soft_unicode(s):
    """Make a string unicode if it isn't already.  That way a markup
    string is not converted back to unicode.
    """
    if not isinstance(s, text_type):
        s = text_type(s)
    return s
