# -*- coding: utf-8 -*-
"""
    jinja2.tests
    ~~~~~~~~~~~~

    Jinja test functions. Used with the "is" operator.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import re
from collections import Mapping
from jinja2.runtime import Undefined
from jinja2._compat import text_type, string_types, integer_types
import decimal

number_re = re.compile(r'^-?\d+(\.\d+)?$')
regex_type = type(number_re)


test_callable = callable


def test_odd(value):
    """Return true if the variable is odd."""
    return value % 2 == 1


def test_even(value):
    """Return true if the variable is even."""
    return value % 2 == 0


def test_divisibleby(value, num):
    """Check if a variable is divisible by a number."""
    return value % num == 0


def test_defined(value):
    """Return true if the variable is defined:

    .. sourcecode:: jinja

        {% if variable is defined %}
            value of variable: {{ variable }}
        {% else %}
            variable is not defined
        {% endif %}

    See the :func:`default` filter for a simple way to set undefined
    variables.
    """
    return not isinstance(value, Undefined)


def test_undefined(value):
    """Like :func:`defined` but the other way round."""
    return isinstance(value, Undefined)


def test_none(value):
    """Return true if the variable is none."""
    return value is None


def test_lower(value):
    """Return true if the variable is lowercased."""
    return text_type(value).islower()


def test_upper(value):
    """Return true if the variable is uppercased."""
    return text_type(value).isupper()


def test_string(value):
    """Return true if the object is a string."""
    return isinstance(value, string_types)


def test_mapping(value):
    """Return true if the object is a mapping (dict etc.).

    .. versionadded:: 2.6
    """
    return isinstance(value, Mapping)


def test_number(value):
    """Return true if the variable is a number."""
    return isinstance(value, integer_types + (float, complex, decimal.Decimal))


def test_sequence(value):
    """Return true if the variable is a sequence. Sequences are variables
    that are iterable.
    """
    try:
        len(value)
        value.__getitem__
    except:
        return False
    return True


def test_equalto(value, other):
    """Check if an object has the same value as another object:

    .. sourcecode:: jinja

        {% if foo.expression is equalto 42 %}
            the foo attribute evaluates to the constant 42
        {% endif %}

    This appears to be a useless test as it does exactly the same as the
    ``==`` operator, but it can be useful when used together with the
    `selectattr` function:

    .. sourcecode:: jinja

        {{ users|selectattr("email", "equalto", "foo@bar.invalid") }}

    .. versionadded:: 2.8
    """
    return value == other


def test_sameas(value, other):
    """Check if an object points to the same memory address than another
    object:

    .. sourcecode:: jinja

        {% if foo.attribute is sameas false %}
            the foo attribute really is the `False` singleton
        {% endif %}
    """
    return value is other


def test_iterable(value):
    """Check if it's possible to iterate over an object."""
    try:
        iter(value)
    except TypeError:
        return False
    return True


def test_escaped(value):
    """Check if the value is escaped."""
    return hasattr(value, '__html__')


def test_greaterthan(value, other):
    """Check if value is greater than other."""
    return value > other


def test_lessthan(value, other):
    """Check if value is less than other."""
    return value < other


TESTS = {
    'odd':              test_odd,
    'even':             test_even,
    'divisibleby':      test_divisibleby,
    'defined':          test_defined,
    'undefined':        test_undefined,
    'none':             test_none,
    'lower':            test_lower,
    'upper':            test_upper,
    'string':           test_string,
    'mapping':          test_mapping,
    'number':           test_number,
    'sequence':         test_sequence,
    'iterable':         test_iterable,
    'callable':         test_callable,
    'sameas':           test_sameas,
    'equalto':          test_equalto,
    'escaped':          test_escaped,
    'greaterthan':      test_greaterthan,
    'lessthan':         test_lessthan
}
