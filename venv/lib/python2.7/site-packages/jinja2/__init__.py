# -*- coding: utf-8 -*-
"""
    jinja2
    ~~~~~~

    Jinja2 is a template engine written in pure Python.  It provides a
    Django inspired non-XML syntax but supports inline expressions and
    an optional sandboxed environment.

    Nutshell
    --------

    Here a small example of a Jinja2 template::

        {% extends 'base.html' %}
        {% block title %}Memberlist{% endblock %}
        {% block content %}
          <ul>
          {% for user in users %}
            <li><a href="{{ user.url }}">{{ user.username }}</a></li>
          {% endfor %}
          </ul>
        {% endblock %}


    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
__docformat__ = 'restructuredtext en'
__version__ = '2.9.5'

# high level interface
from jinja2.environment import Environment, Template

# loaders
from jinja2.loaders import BaseLoader, FileSystemLoader, PackageLoader, \
     DictLoader, FunctionLoader, PrefixLoader, ChoiceLoader, \
     ModuleLoader

# bytecode caches
from jinja2.bccache import BytecodeCache, FileSystemBytecodeCache, \
     MemcachedBytecodeCache

# undefined types
from jinja2.runtime import Undefined, DebugUndefined, StrictUndefined, \
     make_logging_undefined

# exceptions
from jinja2.exceptions import TemplateError, UndefinedError, \
     TemplateNotFound, TemplatesNotFound, TemplateSyntaxError, \
     TemplateAssertionError

# decorators and public utilities
from jinja2.filters import environmentfilter, contextfilter, \
     evalcontextfilter
from jinja2.utils import Markup, escape, clear_caches, \
     environmentfunction, evalcontextfunction, contextfunction, \
     is_undefined, select_autoescape

__all__ = [
    'Environment', 'Template', 'BaseLoader', 'FileSystemLoader',
    'PackageLoader', 'DictLoader', 'FunctionLoader', 'PrefixLoader',
    'ChoiceLoader', 'BytecodeCache', 'FileSystemBytecodeCache',
    'MemcachedBytecodeCache', 'Undefined', 'DebugUndefined',
    'StrictUndefined', 'TemplateError', 'UndefinedError', 'TemplateNotFound',
    'TemplatesNotFound', 'TemplateSyntaxError', 'TemplateAssertionError',
    'ModuleLoader', 'environmentfilter', 'contextfilter', 'Markup', 'escape',
    'environmentfunction', 'contextfunction', 'clear_caches', 'is_undefined',
    'evalcontextfilter', 'evalcontextfunction', 'make_logging_undefined',
    'select_autoescape',
]


def _patch_async():
    from jinja2.utils import have_async_gen
    if have_async_gen:
        from jinja2.asyncsupport import patch_all
        patch_all()


_patch_async()
del _patch_async
