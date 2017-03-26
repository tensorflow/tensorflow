# -*- coding: utf-8 -*-
"""
    jinja2.meta
    ~~~~~~~~~~~

    This module implements various functions that exposes information about
    templates that might be interesting for various kinds of applications.

    :copyright: (c) 2017 by the Jinja Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from jinja2 import nodes
from jinja2.compiler import CodeGenerator
from jinja2._compat import string_types, iteritems


class TrackingCodeGenerator(CodeGenerator):
    """We abuse the code generator for introspection."""

    def __init__(self, environment):
        CodeGenerator.__init__(self, environment, '<introspection>',
                               '<introspection>')
        self.undeclared_identifiers = set()

    def write(self, x):
        """Don't write."""

    def enter_frame(self, frame):
        """Remember all undeclared identifiers."""
        CodeGenerator.enter_frame(self, frame)
        for _, (action, param) in iteritems(frame.symbols.loads):
            if action == 'resolve':
                self.undeclared_identifiers.add(param)


def find_undeclared_variables(ast):
    """Returns a set of all variables in the AST that will be looked up from
    the context at runtime.  Because at compile time it's not known which
    variables will be used depending on the path the execution takes at
    runtime, all variables are returned.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% set foo = 42 %}{{ bar + foo }}')
    >>> meta.find_undeclared_variables(ast) == set(['bar'])
    True

    .. admonition:: Implementation

       Internally the code generator is used for finding undeclared variables.
       This is good to know because the code generator might raise a
       :exc:`TemplateAssertionError` during compilation and as a matter of
       fact this function can currently raise that exception as well.
    """
    codegen = TrackingCodeGenerator(ast.environment)
    codegen.visit(ast)
    return codegen.undeclared_identifiers


def find_referenced_templates(ast):
    """Finds all the referenced templates from the AST.  This will return an
    iterator over all the hardcoded template extensions, inclusions and
    imports.  If dynamic inheritance or inclusion is used, `None` will be
    yielded.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% extends "layout.html" %}{% include helper %}')
    >>> list(meta.find_referenced_templates(ast))
    ['layout.html', None]

    This function is useful for dependency tracking.  For example if you want
    to rebuild parts of the website after a layout template has changed.
    """
    for node in ast.find_all((nodes.Extends, nodes.FromImport, nodes.Import,
                              nodes.Include)):
        if not isinstance(node.template, nodes.Const):
            # a tuple with some non consts in there
            if isinstance(node.template, (nodes.Tuple, nodes.List)):
                for template_name in node.template.items:
                    # something const, only yield the strings and ignore
                    # non-string consts that really just make no sense
                    if isinstance(template_name, nodes.Const):
                        if isinstance(template_name.value, string_types):
                            yield template_name.value
                    # something dynamic in there
                    else:
                        yield None
            # something dynamic we don't know about here
            else:
                yield None
            continue
        # constant is a basestring, direct template name
        if isinstance(node.template.value, string_types):
            yield node.template.value
        # a tuple or list (latter *should* not happen) made of consts,
        # yield the consts that are strings.  We could warn here for
        # non string values
        elif isinstance(node, nodes.Include) and \
             isinstance(node.template.value, (tuple, list)):
            for template_name in node.template.value:
                if isinstance(template_name, string_types):
                    yield template_name
        # something else we don't care about, we could warn here
        else:
            yield None
