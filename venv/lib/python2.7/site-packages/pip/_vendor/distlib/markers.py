# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2013 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
"""Parser for the environment markers micro-language defined in PEP 345."""

import ast
import os
import sys
import platform

from .compat import python_implementation, string_types
from .util import in_venv

__all__ = ['interpret']


class Evaluator(object):
    """
    A limited evaluator for Python expressions.
    """

    operators = {
        'eq': lambda x, y: x == y,
        'gt': lambda x, y: x > y,
        'gte': lambda x, y: x >= y,
        'in': lambda x, y: x in y,
        'lt': lambda x, y: x < y,
        'lte': lambda x, y: x <= y,
        'not': lambda x: not x,
        'noteq': lambda x, y: x != y,
        'notin': lambda x, y: x not in y,
    }

    allowed_values = {
        'sys_platform': sys.platform,
        'python_version': '%s.%s' % sys.version_info[:2],
        # parsing sys.platform is not reliable, but there is no other
        # way to get e.g. 2.7.2+, and the PEP is defined with sys.version
        'python_full_version': sys.version.split(' ', 1)[0],
        'os_name': os.name,
        'platform_in_venv': str(in_venv()),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'platform_machine': platform.machine(),
        'platform_python_implementation': python_implementation(),
    }

    def __init__(self, context=None):
        """
        Initialise an instance.

        :param context: If specified, names are looked up in this mapping.
        """
        self.context = context or {}
        self.source = None

    def get_fragment(self, offset):
        """
        Get the part of the source which is causing a problem.
        """
        fragment_len = 10
        s = '%r' % (self.source[offset:offset + fragment_len])
        if offset + fragment_len < len(self.source):
            s += '...'
        return s

    def get_handler(self, node_type):
        """
        Get a handler for the specified AST node type.
        """
        return getattr(self, 'do_%s' % node_type, None)

    def evaluate(self, node, filename=None):
        """
        Evaluate a source string or node, using ``filename`` when
        displaying errors.
        """
        if isinstance(node, string_types):
            self.source = node
            kwargs = {'mode': 'eval'}
            if filename:
                kwargs['filename'] = filename
            try:
                node = ast.parse(node, **kwargs)
            except SyntaxError as e:
                s = self.get_fragment(e.offset)
                raise SyntaxError('syntax error %s' % s)
        node_type = node.__class__.__name__.lower()
        handler = self.get_handler(node_type)
        if handler is None:
            if self.source is None:
                s = '(source not available)'
            else:
                s = self.get_fragment(node.col_offset)
            raise SyntaxError("don't know how to evaluate %r %s" % (
                node_type, s))
        return handler(node)

    def get_attr_key(self, node):
        assert isinstance(node, ast.Attribute), 'attribute node expected'
        return '%s.%s' % (node.value.id, node.attr)

    def do_attribute(self, node):
        if not isinstance(node.value, ast.Name):
            valid = False
        else:
            key = self.get_attr_key(node)
            valid = key in self.context or key in self.allowed_values
        if not valid:
            raise SyntaxError('invalid expression: %s' % key)
        if key in self.context:
            result = self.context[key]
        else:
            result = self.allowed_values[key]
        return result

    def do_boolop(self, node):
        result = self.evaluate(node.values[0])
        is_or = node.op.__class__ is ast.Or
        is_and = node.op.__class__ is ast.And
        assert is_or or is_and
        if (is_and and result) or (is_or and not result):
            for n in node.values[1:]:
                result = self.evaluate(n)
                if (is_or and result) or (is_and and not result):
                    break
        return result

    def do_compare(self, node):
        def sanity_check(lhsnode, rhsnode):
            valid = True
            if isinstance(lhsnode, ast.Str) and isinstance(rhsnode, ast.Str):
                valid = False
            #elif (isinstance(lhsnode, ast.Attribute)
            #      and isinstance(rhsnode, ast.Attribute)):
            #    klhs = self.get_attr_key(lhsnode)
            #    krhs = self.get_attr_key(rhsnode)
            #    valid = klhs != krhs
            if not valid:
                s = self.get_fragment(node.col_offset)
                raise SyntaxError('Invalid comparison: %s' % s)

        lhsnode = node.left
        lhs = self.evaluate(lhsnode)
        result = True
        for op, rhsnode in zip(node.ops, node.comparators):
            sanity_check(lhsnode, rhsnode)
            op = op.__class__.__name__.lower()
            if op not in self.operators:
                raise SyntaxError('unsupported operation: %r' % op)
            rhs = self.evaluate(rhsnode)
            result = self.operators[op](lhs, rhs)
            if not result:
                break
            lhs = rhs
            lhsnode = rhsnode
        return result

    def do_expression(self, node):
        return self.evaluate(node.body)

    def do_name(self, node):
        valid = False
        if node.id in self.context:
            valid = True
            result = self.context[node.id]
        elif node.id in self.allowed_values:
            valid = True
            result = self.allowed_values[node.id]
        if not valid:
            raise SyntaxError('invalid expression: %s' % node.id)
        return result

    def do_str(self, node):
        return node.s


def interpret(marker, execution_context=None):
    """
    Interpret a marker and return a result depending on environment.

    :param marker: The marker to interpret.
    :type marker: str
    :param execution_context: The context used for name lookup.
    :type execution_context: mapping
    """
    return Evaluator(execution_context).evaluate(marker.strip())
