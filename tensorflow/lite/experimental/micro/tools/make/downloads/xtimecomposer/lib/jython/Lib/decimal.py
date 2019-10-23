# Copyright (c) 2004 Python Software Foundation.
# All rights reserved.

# Written by Eric Price <eprice at tjhsst.edu>
#    and Facundo Batista <facundo at taniquetil.com.ar>
#    and Raymond Hettinger <python at rcn.com>
#    and Aahz <aahz at pobox.com>
#    and Tim Peters

# This module is currently Py2.3 compatible and should be kept that way
# unless a major compelling advantage arises.  IOW, 2.3 compatibility is
# strongly preferred, but not guaranteed.

# Also, this module should be kept in sync with the latest updates of
# the IBM specification as it evolves.  Those updates will be treated
# as bug fixes (deviation from the spec is a compatibility, usability
# bug) and will be backported.  At this point the spec is stabilizing
# and the updates are becoming fewer, smaller, and less significant.

"""
This is a Py2.3 implementation of decimal floating point arithmetic based on
the General Decimal Arithmetic Specification:

    www2.hursley.ibm.com/decimal/decarith.html

and IEEE standard 854-1987:

    www.cs.berkeley.edu/~ejr/projects/754/private/drafts/854-1987/dir.html

Decimal floating point has finite precision with arbitrarily large bounds.

The purpose of this module is to support arithmetic using familiar
"schoolhouse" rules and to avoid some of the tricky representation
issues associated with binary floating point.  The package is especially
useful for financial applications or for contexts where users have
expectations that are at odds with binary floating point (for instance,
in binary floating point, 1.00 % 0.1 gives 0.09999999999999995 instead
of the expected Decimal("0.00") returned by decimal floating point).

Here are some examples of using the decimal module:

>>> from decimal import *
>>> setcontext(ExtendedContext)
>>> Decimal(0)
Decimal("0")
>>> Decimal("1")
Decimal("1")
>>> Decimal("-.0123")
Decimal("-0.0123")
>>> Decimal(123456)
Decimal("123456")
>>> Decimal("123.45e12345678901234567890")
Decimal("1.2345E+12345678901234567892")
>>> Decimal("1.33") + Decimal("1.27")
Decimal("2.60")
>>> Decimal("12.34") + Decimal("3.87") - Decimal("18.41")
Decimal("-2.20")
>>> dig = Decimal(1)
>>> print dig / Decimal(3)
0.333333333
>>> getcontext().prec = 18
>>> print dig / Decimal(3)
0.333333333333333333
>>> print dig.sqrt()
1
>>> print Decimal(3).sqrt()
1.73205080756887729
>>> print Decimal(3) ** 123
4.85192780976896427E+58
>>> inf = Decimal(1) / Decimal(0)
>>> print inf
Infinity
>>> neginf = Decimal(-1) / Decimal(0)
>>> print neginf
-Infinity
>>> print neginf + inf
NaN
>>> print neginf * inf
-Infinity
>>> print dig / 0
Infinity
>>> getcontext().traps[DivisionByZero] = 1
>>> print dig / 0
Traceback (most recent call last):
  ...
  ...
  ...
DivisionByZero: x / 0
>>> c = Context()
>>> c.traps[InvalidOperation] = 0
>>> print c.flags[InvalidOperation]
0
>>> c.divide(Decimal(0), Decimal(0))
Decimal("NaN")
>>> c.traps[InvalidOperation] = 1
>>> print c.flags[InvalidOperation]
1
>>> c.flags[InvalidOperation] = 0
>>> print c.flags[InvalidOperation]
0
>>> print c.divide(Decimal(0), Decimal(0))
Traceback (most recent call last):
  ...
  ...
  ...
InvalidOperation: 0 / 0
>>> print c.flags[InvalidOperation]
1
>>> c.flags[InvalidOperation] = 0
>>> c.traps[InvalidOperation] = 0
>>> print c.divide(Decimal(0), Decimal(0))
NaN
>>> print c.flags[InvalidOperation]
1
>>>
"""

__all__ = [
    # Two major classes
    'Decimal', 'Context',

    # Contexts
    'DefaultContext', 'BasicContext', 'ExtendedContext',

    # Exceptions
    'DecimalException', 'Clamped', 'InvalidOperation', 'DivisionByZero',
    'Inexact', 'Rounded', 'Subnormal', 'Overflow', 'Underflow',

    # Constants for use in setting up contexts
    'ROUND_DOWN', 'ROUND_HALF_UP', 'ROUND_HALF_EVEN', 'ROUND_CEILING',
    'ROUND_FLOOR', 'ROUND_UP', 'ROUND_HALF_DOWN', 'ROUND_05UP',

    # Functions for manipulating contexts
    'setcontext', 'getcontext', 'localcontext'
]

import copy as _copy

# Rounding
ROUND_DOWN = 'ROUND_DOWN'
ROUND_HALF_UP = 'ROUND_HALF_UP'
ROUND_HALF_EVEN = 'ROUND_HALF_EVEN'
ROUND_CEILING = 'ROUND_CEILING'
ROUND_FLOOR = 'ROUND_FLOOR'
ROUND_UP = 'ROUND_UP'
ROUND_HALF_DOWN = 'ROUND_HALF_DOWN'
ROUND_05UP = 'ROUND_05UP'

# Errors

class DecimalException(ArithmeticError):
    """Base exception class.

    Used exceptions derive from this.
    If an exception derives from another exception besides this (such as
    Underflow (Inexact, Rounded, Subnormal) that indicates that it is only
    called if the others are present.  This isn't actually used for
    anything, though.

    handle  -- Called when context._raise_error is called and the
               trap_enabler is set.  First argument is self, second is the
               context.  More arguments can be given, those being after
               the explanation in _raise_error (For example,
               context._raise_error(NewError, '(-x)!', self._sign) would
               call NewError().handle(context, self._sign).)

    To define a new exception, it should be sufficient to have it derive
    from DecimalException.
    """
    def handle(self, context, *args):
        pass


class Clamped(DecimalException):
    """Exponent of a 0 changed to fit bounds.

    This occurs and signals clamped if the exponent of a result has been
    altered in order to fit the constraints of a specific concrete
    representation.  This may occur when the exponent of a zero result would
    be outside the bounds of a representation, or when a large normal
    number would have an encoded exponent that cannot be represented.  In
    this latter case, the exponent is reduced to fit and the corresponding
    number of zero digits are appended to the coefficient ("fold-down").
    """

class InvalidOperation(DecimalException):
    """An invalid operation was performed.

    Various bad things cause this:

    Something creates a signaling NaN
    -INF + INF
    0 * (+-)INF
    (+-)INF / (+-)INF
    x % 0
    (+-)INF % x
    x._rescale( non-integer )
    sqrt(-x) , x > 0
    0 ** 0
    x ** (non-integer)
    x ** (+-)INF
    An operand is invalid

    The result of the operation after these is a quiet positive NaN,
    except when the cause is a signaling NaN, in which case the result is
    also a quiet NaN, but with the original sign, and an optional
    diagnostic information.
    """
    def handle(self, context, *args):
        if args:
            ans = _dec_from_triple(args[0]._sign, args[0]._int, 'n', True)
            return ans._fix_nan(context)
        return NaN

class ConversionSyntax(InvalidOperation):
    """Trying to convert badly formed string.

    This occurs and signals invalid-operation if an string is being
    converted to a number and it does not conform to the numeric string
    syntax.  The result is [0,qNaN].
    """
    def handle(self, context, *args):
        return NaN

class DivisionByZero(DecimalException, ZeroDivisionError):
    """Division by 0.

    This occurs and signals division-by-zero if division of a finite number
    by zero was attempted (during a divide-integer or divide operation, or a
    power operation with negative right-hand operand), and the dividend was
    not zero.

    The result of the operation is [sign,inf], where sign is the exclusive
    or of the signs of the operands for divide, or is 1 for an odd power of
    -0, for power.
    """

    def handle(self, context, sign, *args):
        return Infsign[sign]

class DivisionImpossible(InvalidOperation):
    """Cannot perform the division adequately.

    This occurs and signals invalid-operation if the integer result of a
    divide-integer or remainder operation had too many digits (would be
    longer than precision).  The result is [0,qNaN].
    """

    def handle(self, context, *args):
        return NaN

class DivisionUndefined(InvalidOperation, ZeroDivisionError):
    """Undefined result of division.

    This occurs and signals invalid-operation if division by zero was
    attempted (during a divide-integer, divide, or remainder operation), and
    the dividend is also zero.  The result is [0,qNaN].
    """

    def handle(self, context, *args):
        return NaN

class Inexact(DecimalException):
    """Had to round, losing information.

    This occurs and signals inexact whenever the result of an operation is
    not exact (that is, it needed to be rounded and any discarded digits
    were non-zero), or if an overflow or underflow condition occurs.  The
    result in all cases is unchanged.

    The inexact signal may be tested (or trapped) to determine if a given
    operation (or sequence of operations) was inexact.
    """

class InvalidContext(InvalidOperation):
    """Invalid context.  Unknown rounding, for example.

    This occurs and signals invalid-operation if an invalid context was
    detected during an operation.  This can occur if contexts are not checked
    on creation and either the precision exceeds the capability of the
    underlying concrete representation or an unknown or unsupported rounding
    was specified.  These aspects of the context need only be checked when
    the values are required to be used.  The result is [0,qNaN].
    """

    def handle(self, context, *args):
        return NaN

class Rounded(DecimalException):
    """Number got rounded (not  necessarily changed during rounding).

    This occurs and signals rounded whenever the result of an operation is
    rounded (that is, some zero or non-zero digits were discarded from the
    coefficient), or if an overflow or underflow condition occurs.  The
    result in all cases is unchanged.

    The rounded signal may be tested (or trapped) to determine if a given
    operation (or sequence of operations) caused a loss of precision.
    """

class Subnormal(DecimalException):
    """Exponent < Emin before rounding.

    This occurs and signals subnormal whenever the result of a conversion or
    operation is subnormal (that is, its adjusted exponent is less than
    Emin, before any rounding).  The result in all cases is unchanged.

    The subnormal signal may be tested (or trapped) to determine if a given
    or operation (or sequence of operations) yielded a subnormal result.
    """

class Overflow(Inexact, Rounded):
    """Numerical overflow.

    This occurs and signals overflow if the adjusted exponent of a result
    (from a conversion or from an operation that is not an attempt to divide
    by zero), after rounding, would be greater than the largest value that
    can be handled by the implementation (the value Emax).

    The result depends on the rounding mode:

    For round-half-up and round-half-even (and for round-half-down and
    round-up, if implemented), the result of the operation is [sign,inf],
    where sign is the sign of the intermediate result.  For round-down, the
    result is the largest finite number that can be represented in the
    current precision, with the sign of the intermediate result.  For
    round-ceiling, the result is the same as for round-down if the sign of
    the intermediate result is 1, or is [0,inf] otherwise.  For round-floor,
    the result is the same as for round-down if the sign of the intermediate
    result is 0, or is [1,inf] otherwise.  In all cases, Inexact and Rounded
    will also be raised.
    """

    def handle(self, context, sign, *args):
        if context.rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN,
                                ROUND_HALF_DOWN, ROUND_UP):
            return Infsign[sign]
        if sign == 0:
            if context.rounding == ROUND_CEILING:
                return Infsign[sign]
            return _dec_from_triple(sign, '9'*context.prec,
                            context.Emax-context.prec+1)
        if sign == 1:
            if context.rounding == ROUND_FLOOR:
                return Infsign[sign]
            return _dec_from_triple(sign, '9'*context.prec,
                             context.Emax-context.prec+1)


class Underflow(Inexact, Rounded, Subnormal):
    """Numerical underflow with result rounded to 0.

    This occurs and signals underflow if a result is inexact and the
    adjusted exponent of the result would be smaller (more negative) than
    the smallest value that can be handled by the implementation (the value
    Emin).  That is, the result is both inexact and subnormal.

    The result after an underflow will be a subnormal number rounded, if
    necessary, so that its exponent is not less than Etiny.  This may result
    in 0 with the sign of the intermediate result and an exponent of Etiny.

    In all cases, Inexact, Rounded, and Subnormal will also be raised.
    """

# List of public traps and flags
_signals = [Clamped, DivisionByZero, Inexact, Overflow, Rounded,
           Underflow, InvalidOperation, Subnormal]

# Map conditions (per the spec) to signals
_condition_map = {ConversionSyntax:InvalidOperation,
                  DivisionImpossible:InvalidOperation,
                  DivisionUndefined:InvalidOperation,
                  InvalidContext:InvalidOperation}

##### Context Functions ##################################################

# The getcontext() and setcontext() function manage access to a thread-local
# current context.  Py2.4 offers direct support for thread locals.  If that
# is not available, use threading.currentThread() which is slower but will
# work for older Pythons.  If threads are not part of the build, create a
# mock threading object with threading.local() returning the module namespace.

try:
    import threading
except ImportError:
    # Python was compiled without threads; create a mock object instead
    import sys
    class MockThreading(object):
        def local(self, sys=sys):
            return sys.modules[__name__]
    threading = MockThreading()
    del sys, MockThreading

try:
    threading.local

except AttributeError:

    # To fix reloading, force it to create a new context
    # Old contexts have different exceptions in their dicts, making problems.
    if hasattr(threading.currentThread(), '__decimal_context__'):
        del threading.currentThread().__decimal_context__

    def setcontext(context):
        """Set this thread's context to context."""
        if context in (DefaultContext, BasicContext, ExtendedContext):
            context = context.copy()
            context.clear_flags()
        threading.currentThread().__decimal_context__ = context

    def getcontext():
        """Returns this thread's context.

        If this thread does not yet have a context, returns
        a new context and sets this thread's context.
        New contexts are copies of DefaultContext.
        """
        try:
            return threading.currentThread().__decimal_context__
        except AttributeError:
            context = Context()
            threading.currentThread().__decimal_context__ = context
            return context

else:

    local = threading.local()
    if hasattr(local, '__decimal_context__'):
        del local.__decimal_context__

    def getcontext(_local=local):
        """Returns this thread's context.

        If this thread does not yet have a context, returns
        a new context and sets this thread's context.
        New contexts are copies of DefaultContext.
        """
        try:
            return _local.__decimal_context__
        except AttributeError:
            context = Context()
            _local.__decimal_context__ = context
            return context

    def setcontext(context, _local=local):
        """Set this thread's context to context."""
        if context in (DefaultContext, BasicContext, ExtendedContext):
            context = context.copy()
            context.clear_flags()
        _local.__decimal_context__ = context

    del threading, local        # Don't contaminate the namespace

def localcontext(ctx=None):
    """Return a context manager for a copy of the supplied context

    Uses a copy of the current context if no context is specified
    The returned context manager creates a local decimal context
    in a with statement:
        def sin(x):
             with localcontext() as ctx:
                 ctx.prec += 2
                 # Rest of sin calculation algorithm
                 # uses a precision 2 greater than normal
             return +s  # Convert result to normal precision

         def sin(x):
             with localcontext(ExtendedContext):
                 # Rest of sin calculation algorithm
                 # uses the Extended Context from the
                 # General Decimal Arithmetic Specification
             return +s  # Convert result to normal context

    """
    # The string below can't be included in the docstring until Python 2.6
    # as the doctest module doesn't understand __future__ statements
    """
    >>> from __future__ import with_statement
    >>> print getcontext().prec
    28
    >>> with localcontext():
    ...     ctx = getcontext()
    ...     ctx.prec += 2
    ...     print ctx.prec
    ...
    30
    >>> with localcontext(ExtendedContext):
    ...     print getcontext().prec
    ...
    9
    >>> print getcontext().prec
    28
    """
    if ctx is None: ctx = getcontext()
    return _ContextManager(ctx)


##### Decimal class #######################################################

class Decimal(object):
    """Floating point class for decimal arithmetic."""

    __slots__ = ('_exp','_int','_sign', '_is_special')
    # Generally, the value of the Decimal instance is given by
    #  (-1)**_sign * _int * 10**_exp
    # Special values are signified by _is_special == True

    # We're immutable, so use __new__ not __init__
    def __new__(cls, value="0", context=None):
        """Create a decimal point instance.

        >>> Decimal('3.14')              # string input
        Decimal("3.14")
        >>> Decimal((0, (3, 1, 4), -2))  # tuple (sign, digit_tuple, exponent)
        Decimal("3.14")
        >>> Decimal(314)                 # int or long
        Decimal("314")
        >>> Decimal(Decimal(314))        # another decimal instance
        Decimal("314")
        """

        # Note that the coefficient, self._int, is actually stored as
        # a string rather than as a tuple of digits.  This speeds up
        # the "digits to integer" and "integer to digits" conversions
        # that are used in almost every arithmetic operation on
        # Decimals.  This is an internal detail: the as_tuple function
        # and the Decimal constructor still deal with tuples of
        # digits.

        self = object.__new__(cls)

        # From a string
        # REs insist on real strings, so we can too.
        if isinstance(value, basestring):
            m = _parser(value)
            if m is None:
                if context is None:
                    context = getcontext()
                return context._raise_error(ConversionSyntax,
                                "Invalid literal for Decimal: %r" % value)

            if m.group('sign') == "-":
                self._sign = 1
            else:
                self._sign = 0
            intpart = m.group('int')
            if intpart is not None:
                # finite number
                fracpart = m.group('frac')
                exp = int(m.group('exp') or '0')
                if fracpart is not None:
                    self._int = str((intpart+fracpart).lstrip('0') or '0')
                    self._exp = exp - len(fracpart)
                else:
                    self._int = str(intpart.lstrip('0') or '0')
                    self._exp = exp
                self._is_special = False
            else:
                diag = m.group('diag')
                if diag is not None:
                    # NaN
                    self._int = str(diag.lstrip('0'))
                    if m.group('signal'):
                        self._exp = 'N'
                    else:
                        self._exp = 'n'
                else:
                    # infinity
                    self._int = '0'
                    self._exp = 'F'
                self._is_special = True
            return self

        # From an integer
        if isinstance(value, (int,long)):
            if value >= 0:
                self._sign = 0
            else:
                self._sign = 1
            self._exp = 0
            self._int = str(abs(value))
            self._is_special = False
            return self

        # From another decimal
        if isinstance(value, Decimal):
            self._exp  = value._exp
            self._sign = value._sign
            self._int  = value._int
            self._is_special  = value._is_special
            return self

        # From an internal working value
        if isinstance(value, _WorkRep):
            self._sign = value.sign
            self._int = str(value.int)
            self._exp = int(value.exp)
            self._is_special = False
            return self

        # tuple/list conversion (possibly from as_tuple())
        if isinstance(value, (list,tuple)):
            if len(value) != 3:
                raise ValueError('Invalid tuple size in creation of Decimal '
                                 'from list or tuple.  The list or tuple '
                                 'should have exactly three elements.')
            # process sign.  The isinstance test rejects floats
            if not (isinstance(value[0], (int, long)) and value[0] in (0,1)):
                raise ValueError("Invalid sign.  The first value in the tuple "
                                 "should be an integer; either 0 for a "
                                 "positive number or 1 for a negative number.")
            self._sign = value[0]
            if value[2] == 'F':
                # infinity: value[1] is ignored
                self._int = '0'
                self._exp = value[2]
                self._is_special = True
            else:
                # process and validate the digits in value[1]
                digits = []
                for digit in value[1]:
                    if isinstance(digit, (int, long)) and 0 <= digit <= 9:
                        # skip leading zeros
                        if digits or digit != 0:
                            digits.append(digit)
                    else:
                        raise ValueError("The second value in the tuple must "
                                         "be composed of integers in the range "
                                         "0 through 9.")
                if value[2] in ('n', 'N'):
                    # NaN: digits form the diagnostic
                    self._int = ''.join(map(str, digits))
                    self._exp = value[2]
                    self._is_special = True
                elif isinstance(value[2], (int, long)):
                    # finite number: digits give the coefficient
                    self._int = ''.join(map(str, digits or [0]))
                    self._exp = value[2]
                    self._is_special = False
                else:
                    raise ValueError("The third value in the tuple must "
                                     "be an integer, or one of the "
                                     "strings 'F', 'n', 'N'.")
            return self

        if isinstance(value, float):
            raise TypeError("Cannot convert float to Decimal.  " +
                            "First convert the float to a string")

        raise TypeError("Cannot convert %r to Decimal" % value)

    def _isnan(self):
        """Returns whether the number is not actually one.

        0 if a number
        1 if NaN
        2 if sNaN
        """
        if self._is_special:
            exp = self._exp
            if exp == 'n':
                return 1
            elif exp == 'N':
                return 2
        return 0

    def _isinfinity(self):
        """Returns whether the number is infinite

        0 if finite or not a number
        1 if +INF
        -1 if -INF
        """
        if self._exp == 'F':
            if self._sign:
                return -1
            return 1
        return 0

    def _check_nans(self, other=None, context=None):
        """Returns whether the number is not actually one.

        if self, other are sNaN, signal
        if self, other are NaN return nan
        return 0

        Done before operations.
        """

        self_is_nan = self._isnan()
        if other is None:
            other_is_nan = False
        else:
            other_is_nan = other._isnan()

        if self_is_nan or other_is_nan:
            if context is None:
                context = getcontext()

            if self_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN',
                                        self)
            if other_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN',
                                        other)
            if self_is_nan:
                return self._fix_nan(context)

            return other._fix_nan(context)
        return 0

    def __nonzero__(self):
        """Return True if self is nonzero; otherwise return False.

        NaNs and infinities are considered nonzero.
        """
        return self._is_special or self._int != '0'

    def __cmp__(self, other):
        other = _convert_other(other)
        if other is NotImplemented:
            # Never return NotImplemented
            return 1

        if self._is_special or other._is_special:
            # check for nans, without raising on a signaling nan
            if self._isnan() or other._isnan():
                return 1  # Comparison involving NaN's always reports self > other

            # INF = INF
            return cmp(self._isinfinity(), other._isinfinity())

        # check for zeros;  note that cmp(0, -0) should return 0
        if not self:
            if not other:
                return 0
            else:
                return -((-1)**other._sign)
        if not other:
            return (-1)**self._sign

        # If different signs, neg one is less
        if other._sign < self._sign:
            return -1
        if self._sign < other._sign:
            return 1

        self_adjusted = self.adjusted()
        other_adjusted = other.adjusted()
        if self_adjusted == other_adjusted:
            self_padded = self._int + '0'*(self._exp - other._exp)
            other_padded = other._int + '0'*(other._exp - self._exp)
            return cmp(self_padded, other_padded) * (-1)**self._sign
        elif self_adjusted > other_adjusted:
            return (-1)**self._sign
        else: # self_adjusted < other_adjusted
            return -((-1)**self._sign)

    def __eq__(self, other):
        if not isinstance(other, (Decimal, int, long)):
            return NotImplemented
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        if not isinstance(other, (Decimal, int, long)):
            return NotImplemented
        return self.__cmp__(other) != 0

    def compare(self, other, context=None):
        """Compares one to another.

        -1 => a < b
        0  => a = b
        1  => a > b
        NaN => one is NaN
        Like __cmp__, but returns Decimal instances.
        """
        other = _convert_other(other, raiseit=True)

        # Compare(NaN, NaN) = NaN
        if (self._is_special or other and other._is_special):
            ans = self._check_nans(other, context)
            if ans:
                return ans

        return Decimal(self.__cmp__(other))

    def __hash__(self):
        """x.__hash__() <==> hash(x)"""
        # Decimal integers must hash the same as the ints
        #
        # The hash of a nonspecial noninteger Decimal must depend only
        # on the value of that Decimal, and not on its representation.
        # For example: hash(Decimal("100E-1")) == hash(Decimal("10")).
        if self._is_special:
            if self._isnan():
                raise TypeError('Cannot hash a NaN value.')
            return hash(str(self))
        if not self:
            return 0
        if self._isinteger():
            op = _WorkRep(self.to_integral_value())
            return hash((-1)**op.sign*op.int*10**op.exp)
        # The value of a nonzero nonspecial Decimal instance is
        # faithfully represented by the triple consisting of its sign,
        # its adjusted exponent, and its coefficient with trailing
        # zeros removed.
        return hash((self._sign,
                     self._exp+len(self._int),
                     self._int.rstrip('0')))

    def as_tuple(self):
        """Represents the number as a triple tuple.

        To show the internals exactly as they are.
        """
        return (self._sign, tuple(map(int, self._int)), self._exp)

    def __repr__(self):
        """Represents the number as an instance of Decimal."""
        # Invariant:  eval(repr(d)) == d
        return 'Decimal("%s")' % str(self)

    def __str__(self, eng=False, context=None):
        """Return string representation of the number in scientific notation.

        Captures all of the information in the underlying representation.
        """

        sign = ['', '-'][self._sign]
        if self._is_special:
            if self._exp == 'F':
                return sign + 'Infinity'
            elif self._exp == 'n':
                return sign + 'NaN' + self._int
            else: # self._exp == 'N'
                return sign + 'sNaN' + self._int

        # number of digits of self._int to left of decimal point
        leftdigits = self._exp + len(self._int)

        # dotplace is number of digits of self._int to the left of the
        # decimal point in the mantissa of the output string (that is,
        # after adjusting the exponent)
        if self._exp <= 0 and leftdigits > -6:
            # no exponent required
            dotplace = leftdigits
        elif not eng:
            # usual scientific notation: 1 digit on left of the point
            dotplace = 1
        elif self._int == '0':
            # engineering notation, zero
            dotplace = (leftdigits + 1) % 3 - 1
        else:
            # engineering notation, nonzero
            dotplace = (leftdigits - 1) % 3 + 1

        if dotplace <= 0:
            intpart = '0'
            fracpart = '.' + '0'*(-dotplace) + self._int
        elif dotplace >= len(self._int):
            intpart = self._int+'0'*(dotplace-len(self._int))
            fracpart = ''
        else:
            intpart = self._int[:dotplace]
            fracpart = '.' + self._int[dotplace:]
        if leftdigits == dotplace:
            exp = ''
        else:
            if context is None:
                context = getcontext()
            exp = ['e', 'E'][context.capitals] + "%+d" % (leftdigits-dotplace)

        return sign + intpart + fracpart + exp

    def to_eng_string(self, context=None):
        """Convert to engineering-type string.

        Engineering notation has an exponent which is a multiple of 3, so there
        are up to 3 digits left of the decimal place.

        Same rules for when in exponential and when as a value as in __str__.
        """
        return self.__str__(eng=True, context=context)

    def __neg__(self, context=None):
        """Returns a copy with the sign switched.

        Rounds, if it has reason.
        """
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans

        if not self:
            # -Decimal('0') is Decimal('0'), not Decimal('-0')
            ans = self.copy_abs()
        else:
            ans = self.copy_negate()

        if context is None:
            context = getcontext()
        return ans._fix(context)

    def __pos__(self, context=None):
        """Returns a copy, unless it is a sNaN.

        Rounds the number (if more then precision digits)
        """
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans

        if not self:
            # + (-0) = 0
            ans = self.copy_abs()
        else:
            ans = Decimal(self)

        if context is None:
            context = getcontext()
        return ans._fix(context)

    def __abs__(self, round=True, context=None):
        """Returns the absolute value of self.

        If the keyword argument 'round' is false, do not round.  The
        expression self.__abs__(round=False) is equivalent to
        self.copy_abs().
        """
        if not round:
            return self.copy_abs()

        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans

        if self._sign:
            ans = self.__neg__(context=context)
        else:
            ans = self.__pos__(context=context)

        return ans

    def __add__(self, other, context=None):
        """Returns self + other.

        -INF + INF (or the reverse) cause InvalidOperation errors.
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if context is None:
            context = getcontext()

        if self._is_special or other._is_special:
            ans = self._check_nans(other, context)
            if ans:
                return ans

            if self._isinfinity():
                # If both INF, same sign => same as both, opposite => error.
                if self._sign != other._sign and other._isinfinity():
                    return context._raise_error(InvalidOperation, '-INF + INF')
                return Decimal(self)
            if other._isinfinity():
                return Decimal(other)  # Can't both be infinity here

        exp = min(self._exp, other._exp)
        negativezero = 0
        if context.rounding == ROUND_FLOOR and self._sign != other._sign:
            # If the answer is 0, the sign should be negative, in this case.
            negativezero = 1

        if not self and not other:
            sign = min(self._sign, other._sign)
            if negativezero:
                sign = 1
            ans = _dec_from_triple(sign, '0', exp)
            ans = ans._fix(context)
            return ans
        if not self:
            exp = max(exp, other._exp - context.prec-1)
            ans = other._rescale(exp, context.rounding)
            ans = ans._fix(context)
            return ans
        if not other:
            exp = max(exp, self._exp - context.prec-1)
            ans = self._rescale(exp, context.rounding)
            ans = ans._fix(context)
            return ans

        op1 = _WorkRep(self)
        op2 = _WorkRep(other)
        op1, op2 = _normalize(op1, op2, context.prec)

        result = _WorkRep()
        if op1.sign != op2.sign:
            # Equal and opposite
            if op1.int == op2.int:
                ans = _dec_from_triple(negativezero, '0', exp)
                ans = ans._fix(context)
                return ans
            if op1.int < op2.int:
                op1, op2 = op2, op1
                # OK, now abs(op1) > abs(op2)
            if op1.sign == 1:
                result.sign = 1
                op1.sign, op2.sign = op2.sign, op1.sign
            else:
                result.sign = 0
                # So we know the sign, and op1 > 0.
        elif op1.sign == 1:
            result.sign = 1
            op1.sign, op2.sign = (0, 0)
        else:
            result.sign = 0
        # Now, op1 > abs(op2) > 0

        if op2.sign == 0:
            result.int = op1.int + op2.int
        else:
            result.int = op1.int - op2.int

        result.exp = op1.exp
        ans = Decimal(result)
        ans = ans._fix(context)
        return ans

    __radd__ = __add__

    def __sub__(self, other, context=None):
        """Return self - other"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if self._is_special or other._is_special:
            ans = self._check_nans(other, context=context)
            if ans:
                return ans

        # self - other is computed as self + other.copy_negate()
        return self.__add__(other.copy_negate(), context=context)

    def __rsub__(self, other, context=None):
        """Return other - self"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        return other.__sub__(self, context=context)

    def __mul__(self, other, context=None):
        """Return self * other.

        (+-) INF * 0 (or its reverse) raise InvalidOperation.
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if context is None:
            context = getcontext()

        resultsign = self._sign ^ other._sign

        if self._is_special or other._is_special:
            ans = self._check_nans(other, context)
            if ans:
                return ans

            if self._isinfinity():
                if not other:
                    return context._raise_error(InvalidOperation, '(+-)INF * 0')
                return Infsign[resultsign]

            if other._isinfinity():
                if not self:
                    return context._raise_error(InvalidOperation, '0 * (+-)INF')
                return Infsign[resultsign]

        resultexp = self._exp + other._exp

        # Special case for multiplying by zero
        if not self or not other:
            ans = _dec_from_triple(resultsign, '0', resultexp)
            # Fixing in case the exponent is out of bounds
            ans = ans._fix(context)
            return ans

        # Special case for multiplying by power of 10
        if self._int == '1':
            ans = _dec_from_triple(resultsign, other._int, resultexp)
            ans = ans._fix(context)
            return ans
        if other._int == '1':
            ans = _dec_from_triple(resultsign, self._int, resultexp)
            ans = ans._fix(context)
            return ans

        op1 = _WorkRep(self)
        op2 = _WorkRep(other)

        ans = _dec_from_triple(resultsign, str(op1.int * op2.int), resultexp)
        ans = ans._fix(context)

        return ans
    __rmul__ = __mul__

    def __div__(self, other, context=None):
        """Return self / other."""
        other = _convert_other(other)
        if other is NotImplemented:
            return NotImplemented

        if context is None:
            context = getcontext()

        sign = self._sign ^ other._sign

        if self._is_special or other._is_special:
            ans = self._check_nans(other, context)
            if ans:
                return ans

            if self._isinfinity() and other._isinfinity():
                return context._raise_error(InvalidOperation, '(+-)INF/(+-)INF')

            if self._isinfinity():
                return Infsign[sign]

            if other._isinfinity():
                context._raise_error(Clamped, 'Division by infinity')
                return _dec_from_triple(sign, '0', context.Etiny())

        # Special cases for zeroes
        if not other:
            if not self:
                return context._raise_error(DivisionUndefined, '0 / 0')
            return context._raise_error(DivisionByZero, 'x / 0', sign)

        if not self:
            exp = self._exp - other._exp
            coeff = 0
        else:
            # OK, so neither = 0, INF or NaN
            shift = len(other._int) - len(self._int) + context.prec + 1
            exp = self._exp - other._exp - shift
            op1 = _WorkRep(self)
            op2 = _WorkRep(other)
            if shift >= 0:
                coeff, remainder = divmod(op1.int * 10**shift, op2.int)
            else:
                coeff, remainder = divmod(op1.int, op2.int * 10**-shift)
            if remainder:
                # result is not exact; adjust to ensure correct rounding
                if coeff % 5 == 0:
                    coeff += 1
            else:
                # result is exact; get as close to ideal exponent as possible
                ideal_exp = self._exp - other._exp
                while exp < ideal_exp and coeff % 10 == 0:
                    coeff //= 10
                    exp += 1

        ans = _dec_from_triple(sign, str(coeff), exp)
        return ans._fix(context)

    __truediv__ = __div__

    def _divide(self, other, context):
        """Return (self // other, self % other), to context.prec precision.

        Assumes that neither self nor other is a NaN, that self is not
        infinite and that other is nonzero.
        """
        sign = self._sign ^ other._sign
        if other._isinfinity():
            ideal_exp = self._exp
        else:
            ideal_exp = min(self._exp, other._exp)

        expdiff = self.adjusted() - other.adjusted()
        if not self or other._isinfinity() or expdiff <= -2:
            return (_dec_from_triple(sign, '0', 0),
                    self._rescale(ideal_exp, context.rounding))
        if expdiff <= context.prec:
            op1 = _WorkRep(self)
            op2 = _WorkRep(other)
            if op1.exp >= op2.exp:
                op1.int *= 10**(op1.exp - op2.exp)
            else:
                op2.int *= 10**(op2.exp - op1.exp)
            q, r = divmod(op1.int, op2.int)
            if q < 10**context.prec:
                return (_dec_from_triple(sign, str(q), 0),
                        _dec_from_triple(self._sign, str(r), ideal_exp))

        # Here the quotient is too large to be representable
        ans = context._raise_error(DivisionImpossible,
                                   'quotient too large in //, % or divmod')
        return ans, ans

    def __rdiv__(self, other, context=None):
        """Swaps self/other and returns __div__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__div__(self, context=context)
    __rtruediv__ = __rdiv__

    def __divmod__(self, other, context=None):
        """
        Return (self // other, self % other)
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return (ans, ans)

        sign = self._sign ^ other._sign
        if self._isinfinity():
            if other._isinfinity():
                ans = context._raise_error(InvalidOperation, 'divmod(INF, INF)')
                return ans, ans
            else:
                return (Infsign[sign],
                        context._raise_error(InvalidOperation, 'INF % x'))

        if not other:
            if not self:
                ans = context._raise_error(DivisionUndefined, 'divmod(0, 0)')
                return ans, ans
            else:
                return (context._raise_error(DivisionByZero, 'x // 0', sign),
                        context._raise_error(InvalidOperation, 'x % 0'))

        quotient, remainder = self._divide(other, context)
        remainder = remainder._fix(context)
        return quotient, remainder

    def __rdivmod__(self, other, context=None):
        """Swaps self/other and returns __divmod__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__divmod__(self, context=context)

    def __mod__(self, other, context=None):
        """
        self % other
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return ans

        if self._isinfinity():
            return context._raise_error(InvalidOperation, 'INF % x')
        elif not other:
            if self:
                return context._raise_error(InvalidOperation, 'x % 0')
            else:
                return context._raise_error(DivisionUndefined, '0 % 0')

        remainder = self._divide(other, context)[1]
        remainder = remainder._fix(context)
        return remainder

    def __rmod__(self, other, context=None):
        """Swaps self/other and returns __mod__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__mod__(self, context=context)

    def remainder_near(self, other, context=None):
        """
        Remainder nearest to 0-  abs(remainder-near) <= other/2
        """
        if context is None:
            context = getcontext()

        other = _convert_other(other, raiseit=True)

        ans = self._check_nans(other, context)
        if ans:
            return ans

        # self == +/-infinity -> InvalidOperation
        if self._isinfinity():
            return context._raise_error(InvalidOperation,
                                        'remainder_near(infinity, x)')

        # other == 0 -> either InvalidOperation or DivisionUndefined
        if not other:
            if self:
                return context._raise_error(InvalidOperation,
                                            'remainder_near(x, 0)')
            else:
                return context._raise_error(DivisionUndefined,
                                            'remainder_near(0, 0)')

        # other = +/-infinity -> remainder = self
        if other._isinfinity():
            ans = Decimal(self)
            return ans._fix(context)

        # self = 0 -> remainder = self, with ideal exponent
        ideal_exponent = min(self._exp, other._exp)
        if not self:
            ans = _dec_from_triple(self._sign, '0', ideal_exponent)
            return ans._fix(context)

        # catch most cases of large or small quotient
        expdiff = self.adjusted() - other.adjusted()
        if expdiff >= context.prec + 1:
            # expdiff >= prec+1 => abs(self/other) > 10**prec
            return context._raise_error(DivisionImpossible)
        if expdiff <= -2:
            # expdiff <= -2 => abs(self/other) < 0.1
            ans = self._rescale(ideal_exponent, context.rounding)
            return ans._fix(context)

        # adjust both arguments to have the same exponent, then divide
        op1 = _WorkRep(self)
        op2 = _WorkRep(other)
        if op1.exp >= op2.exp:
            op1.int *= 10**(op1.exp - op2.exp)
        else:
            op2.int *= 10**(op2.exp - op1.exp)
        q, r = divmod(op1.int, op2.int)
        # remainder is r*10**ideal_exponent; other is +/-op2.int *
        # 10**ideal_exponent.   Apply correction to ensure that
        # abs(remainder) <= abs(other)/2
        if 2*r + (q&1) > op2.int:
            r -= op2.int
            q += 1

        if q >= 10**context.prec:
            return context._raise_error(DivisionImpossible)

        # result has same sign as self unless r is negative
        sign = self._sign
        if r < 0:
            sign = 1-sign
            r = -r

        ans = _dec_from_triple(sign, str(r), ideal_exponent)
        return ans._fix(context)

    def __floordiv__(self, other, context=None):
        """self // other"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return ans

        if self._isinfinity():
            if other._isinfinity():
                return context._raise_error(InvalidOperation, 'INF // INF')
            else:
                return Infsign[self._sign ^ other._sign]

        if not other:
            if self:
                return context._raise_error(DivisionByZero, 'x // 0',
                                            self._sign ^ other._sign)
            else:
                return context._raise_error(DivisionUndefined, '0 // 0')

        return self._divide(other, context)[0]

    def __rfloordiv__(self, other, context=None):
        """Swaps self/other and returns __floordiv__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__floordiv__(self, context=context)

    def __float__(self):
        """Float representation."""
        return float(str(self))

    def __int__(self):
        """Converts self to an int, truncating if necessary."""
        if self._is_special:
            if self._isnan():
                context = getcontext()
                return context._raise_error(InvalidContext)
            elif self._isinfinity():
                raise OverflowError("Cannot convert infinity to long")
        s = (-1)**self._sign
        if self._exp >= 0:
            return s*int(self._int)*10**self._exp
        else:
            return s*int(self._int[:self._exp] or '0')

    def __long__(self):
        """Converts to a long.

        Equivalent to long(int(self))
        """
        return long(self.__int__())

    def _fix_nan(self, context):
        """Decapitate the payload of a NaN to fit the context"""
        payload = self._int

        # maximum length of payload is precision if _clamp=0,
        # precision-1 if _clamp=1.
        max_payload_len = context.prec - context._clamp
        if len(payload) > max_payload_len:
            payload = payload[len(payload)-max_payload_len:].lstrip('0')
            return _dec_from_triple(self._sign, payload, self._exp, True)
        return Decimal(self)

    def _fix(self, context):
        """Round if it is necessary to keep self within prec precision.

        Rounds and fixes the exponent.  Does not raise on a sNaN.

        Arguments:
        self - Decimal instance
        context - context used.
        """

        if self._is_special:
            if self._isnan():
                # decapitate payload if necessary
                return self._fix_nan(context)
            else:
                # self is +/-Infinity; return unaltered
                return Decimal(self)

        # if self is zero then exponent should be between Etiny and
        # Emax if _clamp==0, and between Etiny and Etop if _clamp==1.
        Etiny = context.Etiny()
        Etop = context.Etop()
        if not self:
            exp_max = [context.Emax, Etop][context._clamp]
            new_exp = min(max(self._exp, Etiny), exp_max)
            if new_exp != self._exp:
                context._raise_error(Clamped)
                return _dec_from_triple(self._sign, '0', new_exp)
            else:
                return Decimal(self)

        # exp_min is the smallest allowable exponent of the result,
        # equal to max(self.adjusted()-context.prec+1, Etiny)
        exp_min = len(self._int) + self._exp - context.prec
        if exp_min > Etop:
            # overflow: exp_min > Etop iff self.adjusted() > Emax
            context._raise_error(Inexact)
            context._raise_error(Rounded)
            return context._raise_error(Overflow, 'above Emax', self._sign)
        self_is_subnormal = exp_min < Etiny
        if self_is_subnormal:
            context._raise_error(Subnormal)
            exp_min = Etiny

        # round if self has too many digits
        if self._exp < exp_min:
            context._raise_error(Rounded)
            digits = len(self._int) + self._exp - exp_min
            if digits < 0:
                self = _dec_from_triple(self._sign, '1', exp_min-1)
                digits = 0
            this_function = getattr(self, self._pick_rounding_function[context.rounding])
            changed = this_function(digits)
            coeff = self._int[:digits] or '0'
            if changed == 1:
                coeff = str(int(coeff)+1)
            ans = _dec_from_triple(self._sign, coeff, exp_min)

            if changed:
                context._raise_error(Inexact)
                if self_is_subnormal:
                    context._raise_error(Underflow)
                    if not ans:
                        # raise Clamped on underflow to 0
                        context._raise_error(Clamped)
                elif len(ans._int) == context.prec+1:
                    # we get here only if rescaling rounds the
                    # cofficient up to exactly 10**context.prec
                    if ans._exp < Etop:
                        ans = _dec_from_triple(ans._sign,
                                                   ans._int[:-1], ans._exp+1)
                    else:
                        # Inexact and Rounded have already been raised
                        ans = context._raise_error(Overflow, 'above Emax',
                                                   self._sign)
            return ans

        # fold down if _clamp == 1 and self has too few digits
        if context._clamp == 1 and self._exp > Etop:
            context._raise_error(Clamped)
            self_padded = self._int + '0'*(self._exp - Etop)
            return _dec_from_triple(self._sign, self_padded, Etop)

        # here self was representable to begin with; return unchanged
        return Decimal(self)

    _pick_rounding_function = {}

    # for each of the rounding functions below:
    #   self is a finite, nonzero Decimal
    #   prec is an integer satisfying 0 <= prec < len(self._int)
    #
    # each function returns either -1, 0, or 1, as follows:
    #   1 indicates that self should be rounded up (away from zero)
    #   0 indicates that self should be truncated, and that all the
    #     digits to be truncated are zeros (so the value is unchanged)
    #  -1 indicates that there are nonzero digits to be truncated

    def _round_down(self, prec):
        """Also known as round-towards-0, truncate."""
        if _all_zeros(self._int, prec):
            return 0
        else:
            return -1

    def _round_up(self, prec):
        """Rounds away from 0."""
        return -self._round_down(prec)

    def _round_half_up(self, prec):
        """Rounds 5 up (away from 0)"""
        if self._int[prec] in '56789':
            return 1
        elif _all_zeros(self._int, prec):
            return 0
        else:
            return -1

    def _round_half_down(self, prec):
        """Round 5 down"""
        if _exact_half(self._int, prec):
            return -1
        else:
            return self._round_half_up(prec)

    def _round_half_even(self, prec):
        """Round 5 to even, rest to nearest."""
        if _exact_half(self._int, prec) and \
                (prec == 0 or self._int[prec-1] in '02468'):
            return -1
        else:
            return self._round_half_up(prec)

    def _round_ceiling(self, prec):
        """Rounds up (not away from 0 if negative.)"""
        if self._sign:
            return self._round_down(prec)
        else:
            return -self._round_down(prec)

    def _round_floor(self, prec):
        """Rounds down (not towards 0 if negative)"""
        if not self._sign:
            return self._round_down(prec)
        else:
            return -self._round_down(prec)

    def _round_05up(self, prec):
        """Round down unless digit prec-1 is 0 or 5."""
        if prec and self._int[prec-1] not in '05':
            return self._round_down(prec)
        else:
            return -self._round_down(prec)

    def fma(self, other, third, context=None):
        """Fused multiply-add.

        Returns self*other+third with no rounding of the intermediate
        product self*other.

        self and other are multiplied together, with no rounding of
        the result.  The third operand is then added to the result,
        and a single final rounding is performed.
        """

        other = _convert_other(other, raiseit=True)

        # compute product; raise InvalidOperation if either operand is
        # a signaling NaN or if the product is zero times infinity.
        if self._is_special or other._is_special:
            if context is None:
                context = getcontext()
            if self._exp == 'N':
                return context._raise_error(InvalidOperation, 'sNaN', self)
            if other._exp == 'N':
                return context._raise_error(InvalidOperation, 'sNaN', other)
            if self._exp == 'n':
                product = self
            elif other._exp == 'n':
                product = other
            elif self._exp == 'F':
                if not other:
                    return context._raise_error(InvalidOperation,
                                                'INF * 0 in fma')
                product = Infsign[self._sign ^ other._sign]
            elif other._exp == 'F':
                if not self:
                    return context._raise_error(InvalidOperation,
                                                '0 * INF in fma')
                product = Infsign[self._sign ^ other._sign]
        else:
            product = _dec_from_triple(self._sign ^ other._sign,
                                       str(int(self._int) * int(other._int)),
                                       self._exp + other._exp)

        third = _convert_other(third, raiseit=True)
        return product.__add__(third, context)

    def _power_modulo(self, other, modulo, context=None):
        """Three argument version of __pow__"""

        # if can't convert other and modulo to Decimal, raise
        # TypeError; there's no point returning NotImplemented (no
        # equivalent of __rpow__ for three argument pow)
        other = _convert_other(other, raiseit=True)
        modulo = _convert_other(modulo, raiseit=True)

        if context is None:
            context = getcontext()

        # deal with NaNs: if there are any sNaNs then first one wins,
        # (i.e. behaviour for NaNs is identical to that of fma)
        self_is_nan = self._isnan()
        other_is_nan = other._isnan()
        modulo_is_nan = modulo._isnan()
        if self_is_nan or other_is_nan or modulo_is_nan:
            if self_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN',
                                        self)
            if other_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN',
                                        other)
            if modulo_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN',
                                        modulo)
            if self_is_nan:
                return self._fix_nan(context)
            if other_is_nan:
                return other._fix_nan(context)
            return modulo._fix_nan(context)

        # check inputs: we apply same restrictions as Python's pow()
        if not (self._isinteger() and
                other._isinteger() and
                modulo._isinteger()):
            return context._raise_error(InvalidOperation,
                                        'pow() 3rd argument not allowed '
                                        'unless all arguments are integers')
        if other < 0:
            return context._raise_error(InvalidOperation,
                                        'pow() 2nd argument cannot be '
                                        'negative when 3rd argument specified')
        if not modulo:
            return context._raise_error(InvalidOperation,
                                        'pow() 3rd argument cannot be 0')

        # additional restriction for decimal: the modulus must be less
        # than 10**prec in absolute value
        if modulo.adjusted() >= context.prec:
            return context._raise_error(InvalidOperation,
                                        'insufficient precision: pow() 3rd '
                                        'argument must not have more than '
                                        'precision digits')

        # define 0**0 == NaN, for consistency with two-argument pow
        # (even though it hurts!)
        if not other and not self:
            return context._raise_error(InvalidOperation,
                                        'at least one of pow() 1st argument '
                                        'and 2nd argument must be nonzero ;'
                                        '0**0 is not defined')

        # compute sign of result
        if other._iseven():
            sign = 0
        else:
            sign = self._sign

        # convert modulo to a Python integer, and self and other to
        # Decimal integers (i.e. force their exponents to be >= 0)
        modulo = abs(int(modulo))
        base = _WorkRep(self.to_integral_value())
        exponent = _WorkRep(other.to_integral_value())

        # compute result using integer pow()
        base = (base.int % modulo * pow(10, base.exp, modulo)) % modulo
        for i in xrange(exponent.exp):
            base = pow(base, 10, modulo)
        base = pow(base, exponent.int, modulo)

        return _dec_from_triple(sign, str(base), 0)

    def _power_exact(self, other, p):
        """Attempt to compute self**other exactly.

        Given Decimals self and other and an integer p, attempt to
        compute an exact result for the power self**other, with p
        digits of precision.  Return None if self**other is not
        exactly representable in p digits.

        Assumes that elimination of special cases has already been
        performed: self and other must both be nonspecial; self must
        be positive and not numerically equal to 1; other must be
        nonzero.  For efficiency, other._exp should not be too large,
        so that 10**abs(other._exp) is a feasible calculation."""

        # In the comments below, we write x for the value of self and
        # y for the value of other.  Write x = xc*10**xe and y =
        # yc*10**ye.

        # The main purpose of this method is to identify the *failure*
        # of x**y to be exactly representable with as little effort as
        # possible.  So we look for cheap and easy tests that
        # eliminate the possibility of x**y being exact.  Only if all
        # these tests are passed do we go on to actually compute x**y.

        # Here's the main idea.  First normalize both x and y.  We
        # express y as a rational m/n, with m and n relatively prime
        # and n>0.  Then for x**y to be exactly representable (at
        # *any* precision), xc must be the nth power of a positive
        # integer and xe must be divisible by n.  If m is negative
        # then additionally xc must be a power of either 2 or 5, hence
        # a power of 2**n or 5**n.
        #
        # There's a limit to how small |y| can be: if y=m/n as above
        # then:
        #
        #  (1) if xc != 1 then for the result to be representable we
        #      need xc**(1/n) >= 2, and hence also xc**|y| >= 2.  So
        #      if |y| <= 1/nbits(xc) then xc < 2**nbits(xc) <=
        #      2**(1/|y|), hence xc**|y| < 2 and the result is not
        #      representable.
        #
        #  (2) if xe != 0, |xe|*(1/n) >= 1, so |xe|*|y| >= 1.  Hence if
        #      |y| < 1/|xe| then the result is not representable.
        #
        # Note that since x is not equal to 1, at least one of (1) and
        # (2) must apply.  Now |y| < 1/nbits(xc) iff |yc|*nbits(xc) <
        # 10**-ye iff len(str(|yc|*nbits(xc)) <= -ye.
        #
        # There's also a limit to how large y can be, at least if it's
        # positive: the normalized result will have coefficient xc**y,
        # so if it's representable then xc**y < 10**p, and y <
        # p/log10(xc).  Hence if y*log10(xc) >= p then the result is
        # not exactly representable.

        # if len(str(abs(yc*xe)) <= -ye then abs(yc*xe) < 10**-ye,
        # so |y| < 1/xe and the result is not representable.
        # Similarly, len(str(abs(yc)*xc_bits)) <= -ye implies |y|
        # < 1/nbits(xc).

        x = _WorkRep(self)
        xc, xe = x.int, x.exp
        while xc % 10 == 0:
            xc //= 10
            xe += 1

        y = _WorkRep(other)
        yc, ye = y.int, y.exp
        while yc % 10 == 0:
            yc //= 10
            ye += 1

        # case where xc == 1: result is 10**(xe*y), with xe*y
        # required to be an integer
        if xc == 1:
            if ye >= 0:
                exponent = xe*yc*10**ye
            else:
                exponent, remainder = divmod(xe*yc, 10**-ye)
                if remainder:
                    return None
            if y.sign == 1:
                exponent = -exponent
            # if other is a nonnegative integer, use ideal exponent
            if other._isinteger() and other._sign == 0:
                ideal_exponent = self._exp*int(other)
                zeros = min(exponent-ideal_exponent, p-1)
            else:
                zeros = 0
            return _dec_from_triple(0, '1' + '0'*zeros, exponent-zeros)

        # case where y is negative: xc must be either a power
        # of 2 or a power of 5.
        if y.sign == 1:
            last_digit = xc % 10
            if last_digit in (2,4,6,8):
                # quick test for power of 2
                if xc & -xc != xc:
                    return None
                # now xc is a power of 2; e is its exponent
                e = _nbits(xc)-1
                # find e*y and xe*y; both must be integers
                if ye >= 0:
                    y_as_int = yc*10**ye
                    e = e*y_as_int
                    xe = xe*y_as_int
                else:
                    ten_pow = 10**-ye
                    e, remainder = divmod(e*yc, ten_pow)
                    if remainder:
                        return None
                    xe, remainder = divmod(xe*yc, ten_pow)
                    if remainder:
                        return None

                if e*65 >= p*93: # 93/65 > log(10)/log(5)
                    return None
                xc = 5**e

            elif last_digit == 5:
                # e >= log_5(xc) if xc is a power of 5; we have
                # equality all the way up to xc=5**2658
                e = _nbits(xc)*28//65
                xc, remainder = divmod(5**e, xc)
                if remainder:
                    return None
                while xc % 5 == 0:
                    xc //= 5
                    e -= 1
                if ye >= 0:
                    y_as_integer = yc*10**ye
                    e = e*y_as_integer
                    xe = xe*y_as_integer
                else:
                    ten_pow = 10**-ye
                    e, remainder = divmod(e*yc, ten_pow)
                    if remainder:
                        return None
                    xe, remainder = divmod(xe*yc, ten_pow)
                    if remainder:
                        return None
                if e*3 >= p*10: # 10/3 > log(10)/log(2)
                    return None
                xc = 2**e
            else:
                return None

            if xc >= 10**p:
                return None
            xe = -e-xe
            return _dec_from_triple(0, str(xc), xe)

        # now y is positive; find m and n such that y = m/n
        if ye >= 0:
            m, n = yc*10**ye, 1
        else:
            if xe != 0 and len(str(abs(yc*xe))) <= -ye:
                return None
            xc_bits = _nbits(xc)
            if xc != 1 and len(str(abs(yc)*xc_bits)) <= -ye:
                return None
            m, n = yc, 10**(-ye)
            while m % 2 == n % 2 == 0:
                m //= 2
                n //= 2
            while m % 5 == n % 5 == 0:
                m //= 5
                n //= 5

        # compute nth root of xc*10**xe
        if n > 1:
            # if 1 < xc < 2**n then xc isn't an nth power
            if xc != 1 and xc_bits <= n:
                return None

            xe, rem = divmod(xe, n)
            if rem != 0:
                return None

            # compute nth root of xc using Newton's method
            a = 1L << -(-_nbits(xc)//n) # initial estimate
            while True:
                q, r = divmod(xc, a**(n-1))
                if a <= q:
                    break
                else:
                    a = (a*(n-1) + q)//n
            if not (a == q and r == 0):
                return None
            xc = a

        # now xc*10**xe is the nth root of the original xc*10**xe
        # compute mth power of xc*10**xe

        # if m > p*100//_log10_lb(xc) then m > p/log10(xc), hence xc**m >
        # 10**p and the result is not representable.
        if xc > 1 and m > p*100//_log10_lb(xc):
            return None
        xc = xc**m
        xe *= m
        if xc > 10**p:
            return None

        # by this point the result *is* exactly representable
        # adjust the exponent to get as close as possible to the ideal
        # exponent, if necessary
        str_xc = str(xc)
        if other._isinteger() and other._sign == 0:
            ideal_exponent = self._exp*int(other)
            zeros = min(xe-ideal_exponent, p-len(str_xc))
        else:
            zeros = 0
        return _dec_from_triple(0, str_xc+'0'*zeros, xe-zeros)

    def __pow__(self, other, modulo=None, context=None):
        """Return self ** other [ % modulo].

        With two arguments, compute self**other.

        With three arguments, compute (self**other) % modulo.  For the
        three argument form, the following restrictions on the
        arguments hold:

         - all three arguments must be integral
         - other must be nonnegative
         - either self or other (or both) must be nonzero
         - modulo must be nonzero and must have at most p digits,
           where p is the context precision.

        If any of these restrictions is violated the InvalidOperation
        flag is raised.

        The result of pow(self, other, modulo) is identical to the
        result that would be obtained by computing (self**other) %
        modulo with unbounded precision, but is computed more
        efficiently.  It is always exact.
        """

        if modulo is not None:
            return self._power_modulo(other, modulo, context)

        other = _convert_other(other)
        if other is NotImplemented:
            return other

        if context is None:
            context = getcontext()

        # either argument is a NaN => result is NaN
        ans = self._check_nans(other, context)
        if ans:
            return ans

        # 0**0 = NaN (!), x**0 = 1 for nonzero x (including +/-Infinity)
        if not other:
            if not self:
                return context._raise_error(InvalidOperation, '0 ** 0')
            else:
                return Dec_p1

        # result has sign 1 iff self._sign is 1 and other is an odd integer
        result_sign = 0
        if self._sign == 1:
            if other._isinteger():
                if not other._iseven():
                    result_sign = 1
            else:
                # -ve**noninteger = NaN
                # (-0)**noninteger = 0**noninteger
                if self:
                    return context._raise_error(InvalidOperation,
                        'x ** y with x negative and y not an integer')
            # negate self, without doing any unwanted rounding
            self = self.copy_negate()

        # 0**(+ve or Inf)= 0; 0**(-ve or -Inf) = Infinity
        if not self:
            if other._sign == 0:
                return _dec_from_triple(result_sign, '0', 0)
            else:
                return Infsign[result_sign]

        # Inf**(+ve or Inf) = Inf; Inf**(-ve or -Inf) = 0
        if self._isinfinity():
            if other._sign == 0:
                return Infsign[result_sign]
            else:
                return _dec_from_triple(result_sign, '0', 0)

        # 1**other = 1, but the choice of exponent and the flags
        # depend on the exponent of self, and on whether other is a
        # positive integer, a negative integer, or neither
        if self == Dec_p1:
            if other._isinteger():
                # exp = max(self._exp*max(int(other), 0),
                # 1-context.prec) but evaluating int(other) directly
                # is dangerous until we know other is small (other
                # could be 1e999999999)
                if other._sign == 1:
                    multiplier = 0
                elif other > context.prec:
                    multiplier = context.prec
                else:
                    multiplier = int(other)

                exp = self._exp * multiplier
                if exp < 1-context.prec:
                    exp = 1-context.prec
                    context._raise_error(Rounded)
            else:
                context._raise_error(Inexact)
                context._raise_error(Rounded)
                exp = 1-context.prec

            return _dec_from_triple(result_sign, '1'+'0'*-exp, exp)

        # compute adjusted exponent of self
        self_adj = self.adjusted()

        # self ** infinity is infinity if self > 1, 0 if self < 1
        # self ** -infinity is infinity if self < 1, 0 if self > 1
        if other._isinfinity():
            if (other._sign == 0) == (self_adj < 0):
                return _dec_from_triple(result_sign, '0', 0)
            else:
                return Infsign[result_sign]

        # from here on, the result always goes through the call
        # to _fix at the end of this function.
        ans = None

        # crude test to catch cases of extreme overflow/underflow.  If
        # log10(self)*other >= 10**bound and bound >= len(str(Emax))
        # then 10**bound >= 10**len(str(Emax)) >= Emax+1 and hence
        # self**other >= 10**(Emax+1), so overflow occurs.  The test
        # for underflow is similar.
        bound = self._log10_exp_bound() + other.adjusted()
        if (self_adj >= 0) == (other._sign == 0):
            # self > 1 and other +ve, or self < 1 and other -ve
            # possibility of overflow
            if bound >= len(str(context.Emax)):
                ans = _dec_from_triple(result_sign, '1', context.Emax+1)
        else:
            # self > 1 and other -ve, or self < 1 and other +ve
            # possibility of underflow to 0
            Etiny = context.Etiny()
            if bound >= len(str(-Etiny)):
                ans = _dec_from_triple(result_sign, '1', Etiny-1)

        # try for an exact result with precision +1
        if ans is None:
            ans = self._power_exact(other, context.prec + 1)
            if ans is not None and result_sign == 1:
                ans = _dec_from_triple(1, ans._int, ans._exp)

        # usual case: inexact result, x**y computed directly as exp(y*log(x))
        if ans is None:
            p = context.prec
            x = _WorkRep(self)
            xc, xe = x.int, x.exp
            y = _WorkRep(other)
            yc, ye = y.int, y.exp
            if y.sign == 1:
                yc = -yc

            # compute correctly rounded result:  start with precision +3,
            # then increase precision until result is unambiguously roundable
            extra = 3
            while True:
                coeff, exp = _dpower(xc, xe, yc, ye, p+extra)
                if coeff % (5*10**(len(str(coeff))-p-1)):
                    break
                extra += 3

            ans = _dec_from_triple(result_sign, str(coeff), exp)

        # the specification says that for non-integer other we need to
        # raise Inexact, even when the result is actually exact.  In
        # the same way, we need to raise Underflow here if the result
        # is subnormal.  (The call to _fix will take care of raising
        # Rounded and Subnormal, as usual.)
        if not other._isinteger():
            context._raise_error(Inexact)
            # pad with zeros up to length context.prec+1 if necessary
            if len(ans._int) <= context.prec:
                expdiff = context.prec+1 - len(ans._int)
                ans = _dec_from_triple(ans._sign, ans._int+'0'*expdiff,
                                       ans._exp-expdiff)
            if ans.adjusted() < context.Emin:
                context._raise_error(Underflow)

        # unlike exp, ln and log10, the power function respects the
        # rounding mode; no need to use ROUND_HALF_EVEN here
        ans = ans._fix(context)
        return ans

    def __rpow__(self, other, context=None):
        """Swaps self/other and returns __pow__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__pow__(self, context=context)

    def normalize(self, context=None):
        """Normalize- strip trailing 0s, change anything equal to 0 to 0e0"""

        if context is None:
            context = getcontext()

        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans

        dup = self._fix(context)
        if dup._isinfinity():
            return dup

        if not dup:
            return _dec_from_triple(dup._sign, '0', 0)
        exp_max = [context.Emax, context.Etop()][context._clamp]
        end = len(dup._int)
        exp = dup._exp
        while dup._int[end-1] == '0' and exp < exp_max:
            exp += 1
            end -= 1
        return _dec_from_triple(dup._sign, dup._int[:end], exp)

    def quantize(self, exp, rounding=None, context=None, watchexp=True):
        """Quantize self so its exponent is the same as that of exp.

        Similar to self._rescale(exp._exp) but with error checking.
        """
        exp = _convert_other(exp, raiseit=True)

        if context is None:
            context = getcontext()
        if rounding is None:
            rounding = context.rounding

        if self._is_special or exp._is_special:
            ans = self._check_nans(exp, context)
            if ans:
                return ans

            if exp._isinfinity() or self._isinfinity():
                if exp._isinfinity() and self._isinfinity():
                    return Decimal(self)  # if both are inf, it is OK
                return context._raise_error(InvalidOperation,
                                        'quantize with one INF')

        # if we're not watching exponents, do a simple rescale
        if not watchexp:
            ans = self._rescale(exp._exp, rounding)
            # raise Inexact and Rounded where appropriate
            if ans._exp > self._exp:
                context._raise_error(Rounded)
                if ans != self:
                    context._raise_error(Inexact)
            return ans

        # exp._exp should be between Etiny and Emax
        if not (context.Etiny() <= exp._exp <= context.Emax):
            return context._raise_error(InvalidOperation,
                   'target exponent out of bounds in quantize')

        if not self:
            ans = _dec_from_triple(self._sign, '0', exp._exp)
            return ans._fix(context)

        self_adjusted = self.adjusted()
        if self_adjusted > context.Emax:
            return context._raise_error(InvalidOperation,
                                        'exponent of quantize result too large for current context')
        if self_adjusted - exp._exp + 1 > context.prec:
            return context._raise_error(InvalidOperation,
                                        'quantize result has too many digits for current context')

        ans = self._rescale(exp._exp, rounding)
        if ans.adjusted() > context.Emax:
            return context._raise_error(InvalidOperation,
                                        'exponent of quantize result too large for current context')
        if len(ans._int) > context.prec:
            return context._raise_error(InvalidOperation,
                                        'quantize result has too many digits for current context')

        # raise appropriate flags
        if ans._exp > self._exp:
            context._raise_error(Rounded)
            if ans != self:
                context._raise_error(Inexact)
        if ans and ans.adjusted() < context.Emin:
            context._raise_error(Subnormal)

        # call to fix takes care of any necessary folddown
        ans = ans._fix(context)
        return ans

    def same_quantum(self, other):
        """Return True if self and other have the same exponent; otherwise
        return False.

        If either operand is a special value, the following rules are used:
           * return True if both operands are infinities
           * return True if both operands are NaNs
           * otherwise, return False.
        """
        other = _convert_other(other, raiseit=True)
        if self._is_special or other._is_special:
            return (self.is_nan() and other.is_nan() or
                    self.is_infinite() and other.is_infinite())
        return self._exp == other._exp

    def _rescale(self, exp, rounding):
        """Rescale self so that the exponent is exp, either by padding with zeros
        or by truncating digits, using the given rounding mode.

        Specials are returned without change.  This operation is
        quiet: it raises no flags, and uses no information from the
        context.

        exp = exp to scale to (an integer)
        rounding = rounding mode
        """
        if self._is_special:
            return Decimal(self)
        if not self:
            return _dec_from_triple(self._sign, '0', exp)

        if self._exp >= exp:
            # pad answer with zeros if necessary
            return _dec_from_triple(self._sign,
                                        self._int + '0'*(self._exp - exp), exp)

        # too many digits; round and lose data.  If self.adjusted() <
        # exp-1, replace self by 10**(exp-1) before rounding
        digits = len(self._int) + self._exp - exp
        if digits < 0:
            self = _dec_from_triple(self._sign, '1', exp-1)
            digits = 0
        this_function = getattr(self, self._pick_rounding_function[rounding])
        changed = this_function(digits)
        coeff = self._int[:digits] or '0'
        if changed == 1:
            coeff = str(int(coeff)+1)
        return _dec_from_triple(self._sign, coeff, exp)

    def to_integral_exact(self, rounding=None, context=None):
        """Rounds to a nearby integer.

        If no rounding mode is specified, take the rounding mode from
        the context.  This method raises the Rounded and Inexact flags
        when appropriate.

        See also: to_integral_value, which does exactly the same as
        this method except that it doesn't raise Inexact or Rounded.
        """
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
            return Decimal(self)
        if self._exp >= 0:
            return Decimal(self)
        if not self:
            return _dec_from_triple(self._sign, '0', 0)
        if context is None:
            context = getcontext()
        if rounding is None:
            rounding = context.rounding
        context._raise_error(Rounded)
        ans = self._rescale(0, rounding)
        if ans != self:
            context._raise_error(Inexact)
        return ans

    def to_integral_value(self, rounding=None, context=None):
        """Rounds to the nearest integer, without raising inexact, rounded."""
        if context is None:
            context = getcontext()
        if rounding is None:
            rounding = context.rounding
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
            return Decimal(self)
        if self._exp >= 0:
            return Decimal(self)
        else:
            return self._rescale(0, rounding)

    # the method name changed, but we provide also the old one, for compatibility
    to_integral = to_integral_value

    def sqrt(self, context=None):
        """Return the square root of self."""
        if context is None:
            context = getcontext()

        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans

            if self._isinfinity() and self._sign == 0:
                return Decimal(self)

        if not self:
            # exponent = self._exp // 2.  sqrt(-0) = -0
            ans = _dec_from_triple(self._sign, '0', self._exp // 2)
            return ans._fix(context)

        if self._sign == 1:
            return context._raise_error(InvalidOperation, 'sqrt(-x), x > 0')

        # At this point self represents a positive number.  Let p be
        # the desired precision and express self in the form c*100**e
        # with c a positive real number and e an integer, c and e
        # being chosen so that 100**(p-1) <= c < 100**p.  Then the
        # (exact) square root of self is sqrt(c)*10**e, and 10**(p-1)
        # <= sqrt(c) < 10**p, so the closest representable Decimal at
        # precision p is n*10**e where n = round_half_even(sqrt(c)),
        # the closest integer to sqrt(c) with the even integer chosen
        # in the case of a tie.
        #
        # To ensure correct rounding in all cases, we use the
        # following trick: we compute the square root to an extra
        # place (precision p+1 instead of precision p), rounding down.
        # Then, if the result is inexact and its last digit is 0 or 5,
        # we increase the last digit to 1 or 6 respectively; if it's
        # exact we leave the last digit alone.  Now the final round to
        # p places (or fewer in the case of underflow) will round
        # correctly and raise the appropriate flags.

        # use an extra digit of precision
        prec = context.prec+1

        # write argument in the form c*100**e where e = self._exp//2
        # is the 'ideal' exponent, to be used if the square root is
        # exactly representable.  l is the number of 'digits' of c in
        # base 100, so that 100**(l-1) <= c < 100**l.
        op = _WorkRep(self)
        e = op.exp >> 1
        if op.exp & 1:
            c = op.int * 10
            l = (len(self._int) >> 1) + 1
        else:
            c = op.int
            l = len(self._int)+1 >> 1

        # rescale so that c has exactly prec base 100 'digits'
        shift = prec-l
        if shift >= 0:
            c *= 100**shift
            exact = True
        else:
            c, remainder = divmod(c, 100**-shift)
            exact = not remainder
        e -= shift

        # find n = floor(sqrt(c)) using Newton's method
        n = 10**prec
        while True:
            q = c//n
            if n <= q:
                break
            else:
                n = n + q >> 1
        exact = exact and n*n == c

        if exact:
            # result is exact; rescale to use ideal exponent e
            if shift >= 0:
                # assert n % 10**shift == 0
                n //= 10**shift
            else:
                n *= 10**-shift
            e += shift
        else:
            # result is not exact; fix last digit as described above
            if n % 5 == 0:
                n += 1

        ans = _dec_from_triple(0, str(n), e)

        # round, and fit to current context
        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding

        return ans

    def max(self, other, context=None):
        """Returns the larger value.

        Like max(self, other) except if one is not a number, returns
        NaN (and signals if one is sNaN).  Also rounds.
        """
        other = _convert_other(other, raiseit=True)

        if context is None:
            context = getcontext()

        if self._is_special or other._is_special:
            # If one operand is a quiet NaN and the other is number, then the
            # number is always returned
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)

        c = self.__cmp__(other)
        if c == 0:
            # If both operands are finite and equal in numerical value
            # then an ordering is applied:
            #
            # If the signs differ then max returns the operand with the
            # positive sign and min returns the operand with the negative sign
            #
            # If the signs are the same then the exponent is used to select
            # the result.  This is exactly the ordering used in compare_total.
            c = self.compare_total(other)

        if c == -1:
            ans = other
        else:
            ans = self

        return ans._fix(context)

    def min(self, other, context=None):
        """Returns the smaller value.

        Like min(self, other) except if one is not a number, returns
        NaN (and signals if one is sNaN).  Also rounds.
        """
        other = _convert_other(other, raiseit=True)

        if context is None:
            context = getcontext()

        if self._is_special or other._is_special:
            # If one operand is a quiet NaN and the other is number, then the
            # number is always returned
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)

        c = self.__cmp__(other)
        if c == 0:
            c = self.compare_total(other)

        if c == -1:
            ans = self
        else:
            ans = other

        return ans._fix(context)

    def _isinteger(self):
        """Returns whether self is an integer"""
        if self._is_special:
            return False
        if self._exp >= 0:
            return True
        rest = self._int[self._exp:]
        return rest == '0'*len(rest)

    def _iseven(self):
        """Returns True if self is even.  Assumes self is an integer."""
        if not self or self._exp > 0:
            return True
        return self._int[-1+self._exp] in '02468'

    def adjusted(self):
        """Return the adjusted exponent of self"""
        try:
            return self._exp + len(self._int) - 1
        # If NaN or Infinity, self._exp is string
        except TypeError:
            return 0

    def canonical(self, context=None):
        """Returns the same Decimal object.

        As we do not have different encodings for the same number, the
        received object already is in its canonical form.
        """
        return self

    def compare_signal(self, other, context=None):
        """Compares self to the other operand numerically.

        It's pretty much like compare(), but all NaNs signal, with signaling
        NaNs taking precedence over quiet NaNs.
        """
        if context is None:
            context = getcontext()

        self_is_nan = self._isnan()
        other_is_nan = other._isnan()
        if self_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN',
                                        self)
        if other_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN',
                                        other)
        if self_is_nan:
            return context._raise_error(InvalidOperation, 'NaN in compare_signal',
                                        self)
        if other_is_nan:
            return context._raise_error(InvalidOperation, 'NaN in compare_signal',
                                        other)
        return self.compare(other, context=context)

    def compare_total(self, other):
        """Compares self to other using the abstract representations.

        This is not like the standard compare, which use their numerical
        value. Note that a total ordering is defined for all possible abstract
        representations.
        """
        # if one is negative and the other is positive, it's easy
        if self._sign and not other._sign:
            return Dec_n1
        if not self._sign and other._sign:
            return Dec_p1
        sign = self._sign

        # let's handle both NaN types
        self_nan = self._isnan()
        other_nan = other._isnan()
        if self_nan or other_nan:
            if self_nan == other_nan:
                if self._int < other._int:
                    if sign:
                        return Dec_p1
                    else:
                        return Dec_n1
                if self._int > other._int:
                    if sign:
                        return Dec_n1
                    else:
                        return Dec_p1
                return Dec_0

            if sign:
                if self_nan == 1:
                    return Dec_n1
                if other_nan == 1:
                    return Dec_p1
                if self_nan == 2:
                    return Dec_n1
                if other_nan == 2:
                    return Dec_p1
            else:
                if self_nan == 1:
                    return Dec_p1
                if other_nan == 1:
                    return Dec_n1
                if self_nan == 2:
                    return Dec_p1
                if other_nan == 2:
                    return Dec_n1

        if self < other:
            return Dec_n1
        if self > other:
            return Dec_p1

        if self._exp < other._exp:
            if sign:
                return Dec_p1
            else:
                return Dec_n1
        if self._exp > other._exp:
            if sign:
                return Dec_n1
            else:
                return Dec_p1
        return Dec_0


    def compare_total_mag(self, other):
        """Compares self to other using abstract repr., ignoring sign.

        Like compare_total, but with operand's sign ignored and assumed to be 0.
        """
        s = self.copy_abs()
        o = other.copy_abs()
        return s.compare_total(o)

    def copy_abs(self):
        """Returns a copy with the sign set to 0. """
        return _dec_from_triple(0, self._int, self._exp, self._is_special)

    def copy_negate(self):
        """Returns a copy with the sign inverted."""
        if self._sign:
            return _dec_from_triple(0, self._int, self._exp, self._is_special)
        else:
            return _dec_from_triple(1, self._int, self._exp, self._is_special)

    def copy_sign(self, other):
        """Returns self with the sign of other."""
        return _dec_from_triple(other._sign, self._int,
                                self._exp, self._is_special)

    def exp(self, context=None):
        """Returns e ** self."""

        if context is None:
            context = getcontext()

        # exp(NaN) = NaN
        ans = self._check_nans(context=context)
        if ans:
            return ans

        # exp(-Infinity) = 0
        if self._isinfinity() == -1:
            return Dec_0

        # exp(0) = 1
        if not self:
            return Dec_p1

        # exp(Infinity) = Infinity
        if self._isinfinity() == 1:
            return Decimal(self)

        # the result is now guaranteed to be inexact (the true
        # mathematical result is transcendental). There's no need to
        # raise Rounded and Inexact here---they'll always be raised as
        # a result of the call to _fix.
        p = context.prec
        adj = self.adjusted()

        # we only need to do any computation for quite a small range
        # of adjusted exponents---for example, -29 <= adj <= 10 for
        # the default context.  For smaller exponent the result is
        # indistinguishable from 1 at the given precision, while for
        # larger exponent the result either overflows or underflows.
        if self._sign == 0 and adj > len(str((context.Emax+1)*3)):
            # overflow
            ans = _dec_from_triple(0, '1', context.Emax+1)
        elif self._sign == 1 and adj > len(str((-context.Etiny()+1)*3)):
            # underflow to 0
            ans = _dec_from_triple(0, '1', context.Etiny()-1)
        elif self._sign == 0 and adj < -p:
            # p+1 digits; final round will raise correct flags
            ans = _dec_from_triple(0, '1' + '0'*(p-1) + '1', -p)
        elif self._sign == 1 and adj < -p-1:
            # p+1 digits; final round will raise correct flags
            ans = _dec_from_triple(0, '9'*(p+1), -p-1)
        # general case
        else:
            op = _WorkRep(self)
            c, e = op.int, op.exp
            if op.sign == 1:
                c = -c

            # compute correctly rounded result: increase precision by
            # 3 digits at a time until we get an unambiguously
            # roundable result
            extra = 3
            while True:
                coeff, exp = _dexp(c, e, p+extra)
                if coeff % (5*10**(len(str(coeff))-p-1)):
                    break
                extra += 3

            ans = _dec_from_triple(0, str(coeff), exp)

        # at this stage, ans should round correctly with *any*
        # rounding mode, not just with ROUND_HALF_EVEN
        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding

        return ans

    def is_canonical(self):
        """Return True if self is canonical; otherwise return False.

        Currently, the encoding of a Decimal instance is always
        canonical, so this method returns True for any Decimal.
        """
        return True

    def is_finite(self):
        """Return True if self is finite; otherwise return False.

        A Decimal instance is considered finite if it is neither
        infinite nor a NaN.
        """
        return not self._is_special

    def is_infinite(self):
        """Return True if self is infinite; otherwise return False."""
        return self._exp == 'F'

    def is_nan(self):
        """Return True if self is a qNaN or sNaN; otherwise return False."""
        return self._exp in ('n', 'N')

    def is_normal(self, context=None):
        """Return True if self is a normal number; otherwise return False."""
        if self._is_special or not self:
            return False
        if context is None:
            context = getcontext()
        return context.Emin <= self.adjusted() <= context.Emax

    def is_qnan(self):
        """Return True if self is a quiet NaN; otherwise return False."""
        return self._exp == 'n'

    def is_signed(self):
        """Return True if self is negative; otherwise return False."""
        return self._sign == 1

    def is_snan(self):
        """Return True if self is a signaling NaN; otherwise return False."""
        return self._exp == 'N'

    def is_subnormal(self, context=None):
        """Return True if self is subnormal; otherwise return False."""
        if self._is_special or not self:
            return False
        if context is None:
            context = getcontext()
        return self.adjusted() < context.Emin

    def is_zero(self):
        """Return True if self is a zero; otherwise return False."""
        return not self._is_special and self._int == '0'

    def _ln_exp_bound(self):
        """Compute a lower bound for the adjusted exponent of self.ln().
        In other words, compute r such that self.ln() >= 10**r.  Assumes
        that self is finite and positive and that self != 1.
        """

        # for 0.1 <= x <= 10 we use the inequalities 1-1/x <= ln(x) <= x-1
        adj = self._exp + len(self._int) - 1
        if adj >= 1:
            # argument >= 10; we use 23/10 = 2.3 as a lower bound for ln(10)
            return len(str(adj*23//10)) - 1
        if adj <= -2:
            # argument <= 0.1
            return len(str((-1-adj)*23//10)) - 1
        op = _WorkRep(self)
        c, e = op.int, op.exp
        if adj == 0:
            # 1 < self < 10
            num = str(c-10**-e)
            den = str(c)
            return len(num) - len(den) - (num < den)
        # adj == -1, 0.1 <= self < 1
        return e + len(str(10**-e - c)) - 1


    def ln(self, context=None):
        """Returns the natural (base e) logarithm of self."""

        if context is None:
            context = getcontext()

        # ln(NaN) = NaN
        ans = self._check_nans(context=context)
        if ans:
            return ans

        # ln(0.0) == -Infinity
        if not self:
            return negInf

        # ln(Infinity) = Infinity
        if self._isinfinity() == 1:
            return Inf

        # ln(1.0) == 0.0
        if self == Dec_p1:
            return Dec_0

        # ln(negative) raises InvalidOperation
        if self._sign == 1:
            return context._raise_error(InvalidOperation,
                                        'ln of a negative value')

        # result is irrational, so necessarily inexact
        op = _WorkRep(self)
        c, e = op.int, op.exp
        p = context.prec

        # correctly rounded result: repeatedly increase precision by 3
        # until we get an unambiguously roundable result
        places = p - self._ln_exp_bound() + 2 # at least p+3 places
        while True:
            coeff = _dlog(c, e, places)
            # assert len(str(abs(coeff)))-p >= 1
            if coeff % (5*10**(len(str(abs(coeff)))-p-1)):
                break
            places += 3
        ans = _dec_from_triple(int(coeff<0), str(abs(coeff)), -places)

        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding
        return ans

    def _log10_exp_bound(self):
        """Compute a lower bound for the adjusted exponent of self.log10().
        In other words, find r such that self.log10() >= 10**r.
        Assumes that self is finite and positive and that self != 1.
        """

        # For x >= 10 or x < 0.1 we only need a bound on the integer
        # part of log10(self), and this comes directly from the
        # exponent of x.  For 0.1 <= x <= 10 we use the inequalities
        # 1-1/x <= log(x) <= x-1. If x > 1 we have |log10(x)| >
        # (1-1/x)/2.31 > 0.  If x < 1 then |log10(x)| > (1-x)/2.31 > 0

        adj = self._exp + len(self._int) - 1
        if adj >= 1:
            # self >= 10
            return len(str(adj))-1
        if adj <= -2:
            # self < 0.1
            return len(str(-1-adj))-1
        op = _WorkRep(self)
        c, e = op.int, op.exp
        if adj == 0:
            # 1 < self < 10
            num = str(c-10**-e)
            den = str(231*c)
            return len(num) - len(den) - (num < den) + 2
        # adj == -1, 0.1 <= self < 1
        num = str(10**-e-c)
        return len(num) + e - (num < "231") - 1

    def log10(self, context=None):
        """Returns the base 10 logarithm of self."""

        if context is None:
            context = getcontext()

        # log10(NaN) = NaN
        ans = self._check_nans(context=context)
        if ans:
            return ans

        # log10(0.0) == -Infinity
        if not self:
            return negInf

        # log10(Infinity) = Infinity
        if self._isinfinity() == 1:
            return Inf

        # log10(negative or -Infinity) raises InvalidOperation
        if self._sign == 1:
            return context._raise_error(InvalidOperation,
                                        'log10 of a negative value')

        # log10(10**n) = n
        if self._int[0] == '1' and self._int[1:] == '0'*(len(self._int) - 1):
            # answer may need rounding
            ans = Decimal(self._exp + len(self._int) - 1)
        else:
            # result is irrational, so necessarily inexact
            op = _WorkRep(self)
            c, e = op.int, op.exp
            p = context.prec

            # correctly rounded result: repeatedly increase precision
            # until result is unambiguously roundable
            places = p-self._log10_exp_bound()+2
            while True:
                coeff = _dlog10(c, e, places)
                # assert len(str(abs(coeff)))-p >= 1
                if coeff % (5*10**(len(str(abs(coeff)))-p-1)):
                    break
                places += 3
            ans = _dec_from_triple(int(coeff<0), str(abs(coeff)), -places)

        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding
        return ans

    def logb(self, context=None):
        """ Returns the exponent of the magnitude of self's MSD.

        The result is the integer which is the exponent of the magnitude
        of the most significant digit of self (as though it were truncated
        to a single digit while maintaining the value of that digit and
        without limiting the resulting exponent).
        """
        # logb(NaN) = NaN
        ans = self._check_nans(context=context)
        if ans:
            return ans

        if context is None:
            context = getcontext()

        # logb(+/-Inf) = +Inf
        if self._isinfinity():
            return Inf

        # logb(0) = -Inf, DivisionByZero
        if not self:
            return context._raise_error(DivisionByZero, 'logb(0)', 1)

        # otherwise, simply return the adjusted exponent of self, as a
        # Decimal.  Note that no attempt is made to fit the result
        # into the current context.
        return Decimal(self.adjusted())

    def _islogical(self):
        """Return True if self is a logical operand.

        For being logical, it must be a finite numbers with a sign of 0,
        an exponent of 0, and a coefficient whose digits must all be
        either 0 or 1.
        """
        if self._sign != 0 or self._exp != 0:
            return False
        for dig in self._int:
            if dig not in '01':
                return False
        return True

    def _fill_logical(self, context, opa, opb):
        dif = context.prec - len(opa)
        if dif > 0:
            opa = '0'*dif + opa
        elif dif < 0:
            opa = opa[-context.prec:]
        dif = context.prec - len(opb)
        if dif > 0:
            opb = '0'*dif + opb
        elif dif < 0:
            opb = opb[-context.prec:]
        return opa, opb

    def logical_and(self, other, context=None):
        """Applies an 'and' operation between self and other's digits."""
        if context is None:
            context = getcontext()
        if not self._islogical() or not other._islogical():
            return context._raise_error(InvalidOperation)

        # fill to context.prec
        (opa, opb) = self._fill_logical(context, self._int, other._int)

        # make the operation, and clean starting zeroes
        result = "".join([str(int(a)&int(b)) for a,b in zip(opa,opb)])
        return _dec_from_triple(0, result.lstrip('0') or '0', 0)

    def logical_invert(self, context=None):
        """Invert all its digits."""
        if context is None:
            context = getcontext()
        return self.logical_xor(_dec_from_triple(0,'1'*context.prec,0),
                                context)

    def logical_or(self, other, context=None):
        """Applies an 'or' operation between self and other's digits."""
        if context is None:
            context = getcontext()
        if not self._islogical() or not other._islogical():
            return context._raise_error(InvalidOperation)

        # fill to context.prec
        (opa, opb) = self._fill_logical(context, self._int, other._int)

        # make the operation, and clean starting zeroes
        result = "".join(str(int(a)|int(b)) for a,b in zip(opa,opb))
        return _dec_from_triple(0, result.lstrip('0') or '0', 0)

    def logical_xor(self, other, context=None):
        """Applies an 'xor' operation between self and other's digits."""
        if context is None:
            context = getcontext()
        if not self._islogical() or not other._islogical():
            return context._raise_error(InvalidOperation)

        # fill to context.prec
        (opa, opb) = self._fill_logical(context, self._int, other._int)

        # make the operation, and clean starting zeroes
        result = "".join(str(int(a)^int(b)) for a,b in zip(opa,opb))
        return _dec_from_triple(0, result.lstrip('0') or '0', 0)

    def max_mag(self, other, context=None):
        """Compares the values numerically with their sign ignored."""
        other = _convert_other(other, raiseit=True)

        if context is None:
            context = getcontext()

        if self._is_special or other._is_special:
            # If one operand is a quiet NaN and the other is number, then the
            # number is always returned
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)

        c = self.copy_abs().__cmp__(other.copy_abs())
        if c == 0:
            c = self.compare_total(other)

        if c == -1:
            ans = other
        else:
            ans = self

        return ans._fix(context)

    def min_mag(self, other, context=None):
        """Compares the values numerically with their sign ignored."""
        other = _convert_other(other, raiseit=True)

        if context is None:
            context = getcontext()

        if self._is_special or other._is_special:
            # If one operand is a quiet NaN and the other is number, then the
            # number is always returned
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)

        c = self.copy_abs().__cmp__(other.copy_abs())
        if c == 0:
            c = self.compare_total(other)

        if c == -1:
            ans = self
        else:
            ans = other

        return ans._fix(context)

    def next_minus(self, context=None):
        """Returns the largest representable number smaller than itself."""
        if context is None:
            context = getcontext()

        ans = self._check_nans(context=context)
        if ans:
            return ans

        if self._isinfinity() == -1:
            return negInf
        if self._isinfinity() == 1:
            return _dec_from_triple(0, '9'*context.prec, context.Etop())

        context = context.copy()
        context._set_rounding(ROUND_FLOOR)
        context._ignore_all_flags()
        new_self = self._fix(context)
        if new_self != self:
            return new_self
        return self.__sub__(_dec_from_triple(0, '1', context.Etiny()-1),
                            context)

    def next_plus(self, context=None):
        """Returns the smallest representable number larger than itself."""
        if context is None:
            context = getcontext()

        ans = self._check_nans(context=context)
        if ans:
            return ans

        if self._isinfinity() == 1:
            return Inf
        if self._isinfinity() == -1:
            return _dec_from_triple(1, '9'*context.prec, context.Etop())

        context = context.copy()
        context._set_rounding(ROUND_CEILING)
        context._ignore_all_flags()
        new_self = self._fix(context)
        if new_self != self:
            return new_self
        return self.__add__(_dec_from_triple(0, '1', context.Etiny()-1),
                            context)

    def next_toward(self, other, context=None):
        """Returns the number closest to self, in the direction towards other.

        The result is the closest representable number to self
        (excluding self) that is in the direction towards other,
        unless both have the same value.  If the two operands are
        numerically equal, then the result is a copy of self with the
        sign set to be the same as the sign of other.
        """
        other = _convert_other(other, raiseit=True)

        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return ans

        comparison = self.__cmp__(other)
        if comparison == 0:
            return self.copy_sign(other)

        if comparison == -1:
            ans = self.next_plus(context)
        else: # comparison == 1
            ans = self.next_minus(context)

        # decide which flags to raise using value of ans
        if ans._isinfinity():
            context._raise_error(Overflow,
                                 'Infinite result from next_toward',
                                 ans._sign)
            context._raise_error(Rounded)
            context._raise_error(Inexact)
        elif ans.adjusted() < context.Emin:
            context._raise_error(Underflow)
            context._raise_error(Subnormal)
            context._raise_error(Rounded)
            context._raise_error(Inexact)
            # if precision == 1 then we don't raise Clamped for a
            # result 0E-Etiny.
            if not ans:
                context._raise_error(Clamped)

        return ans

    def number_class(self, context=None):
        """Returns an indication of the class of self.

        The class is one of the following strings:
          sNaN
          NaN
          -Infinity
          -Normal
          -Subnormal
          -Zero
          +Zero
          +Subnormal
          +Normal
          +Infinity
        """
        if self.is_snan():
            return "sNaN"
        if self.is_qnan():
            return "NaN"
        inf = self._isinfinity()
        if inf == 1:
            return "+Infinity"
        if inf == -1:
            return "-Infinity"
        if self.is_zero():
            if self._sign:
                return "-Zero"
            else:
                return "+Zero"
        if context is None:
            context = getcontext()
        if self.is_subnormal(context=context):
            if self._sign:
                return "-Subnormal"
            else:
                return "+Subnormal"
        # just a normal, regular, boring number, :)
        if self._sign:
            return "-Normal"
        else:
            return "+Normal"

    def radix(self):
        """Just returns 10, as this is Decimal, :)"""
        return Decimal(10)

    def rotate(self, other, context=None):
        """Returns a rotated copy of self, value-of-other times."""
        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return ans

        if other._exp != 0:
            return context._raise_error(InvalidOperation)
        if not (-context.prec <= int(other) <= context.prec):
            return context._raise_error(InvalidOperation)

        if self._isinfinity():
            return Decimal(self)

        # get values, pad if necessary
        torot = int(other)
        rotdig = self._int
        topad = context.prec - len(rotdig)
        if topad:
            rotdig = '0'*topad + rotdig

        # let's rotate!
        rotated = rotdig[torot:] + rotdig[:torot]
        return _dec_from_triple(self._sign,
                                rotated.lstrip('0') or '0', self._exp)

    def scaleb (self, other, context=None):
        """Returns self operand after adding the second value to its exp."""
        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return ans

        if other._exp != 0:
            return context._raise_error(InvalidOperation)
        liminf = -2 * (context.Emax + context.prec)
        limsup =  2 * (context.Emax + context.prec)
        if not (liminf <= int(other) <= limsup):
            return context._raise_error(InvalidOperation)

        if self._isinfinity():
            return Decimal(self)

        d = _dec_from_triple(self._sign, self._int, self._exp + int(other))
        d = d._fix(context)
        return d

    def shift(self, other, context=None):
        """Returns a shifted copy of self, value-of-other times."""
        if context is None:
            context = getcontext()

        ans = self._check_nans(other, context)
        if ans:
            return ans

        if other._exp != 0:
            return context._raise_error(InvalidOperation)
        if not (-context.prec <= int(other) <= context.prec):
            return context._raise_error(InvalidOperation)

        if self._isinfinity():
            return Decimal(self)

        # get values, pad if necessary
        torot = int(other)
        if not torot:
            return Decimal(self)
        rotdig = self._int
        topad = context.prec - len(rotdig)
        if topad:
            rotdig = '0'*topad + rotdig

        # let's shift!
        if torot < 0:
            rotated = rotdig[:torot]
        else:
            rotated = rotdig + '0'*torot
            rotated = rotated[-context.prec:]

        return _dec_from_triple(self._sign,
                                    rotated.lstrip('0') or '0', self._exp)

    # Support for pickling, copy, and deepcopy
    def __reduce__(self):
        return (self.__class__, (str(self),))

    def __copy__(self):
        if type(self) == Decimal:
            return self     # I'm immutable; therefore I am my own clone
        return self.__class__(str(self))

    def __deepcopy__(self, memo):
        if type(self) == Decimal:
            return self     # My components are also immutable
        return self.__class__(str(self))

    # support for Jython __tojava__:
    def __tojava__(self, java_class):
        from java.lang import Object
        from java.math import BigDecimal
        from org.python.core import Py
        if java_class not in (BigDecimal, Object):
            return Py.NoConversion
        return BigDecimal(str(self))

def _dec_from_triple(sign, coefficient, exponent, special=False):
    """Create a decimal instance directly, without any validation,
    normalization (e.g. removal of leading zeros) or argument
    conversion.

    This function is for *internal use only*.
    """

    self = object.__new__(Decimal)
    self._sign = sign
    self._int = coefficient
    self._exp = exponent
    self._is_special = special

    return self

##### Context class #######################################################


# get rounding method function:
rounding_functions = [name for name in Decimal.__dict__.keys()
                                    if name.startswith('_round_')]
for name in rounding_functions:
    # name is like _round_half_even, goes to the global ROUND_HALF_EVEN value.
    globalname = name[1:].upper()
    val = globals()[globalname]
    Decimal._pick_rounding_function[val] = name

del name, val, globalname, rounding_functions

class _ContextManager(object):
    """Context manager class to support localcontext().

      Sets a copy of the supplied context in __enter__() and restores
      the previous decimal context in __exit__()
    """
    def __init__(self, new_context):
        self.new_context = new_context.copy()
    def __enter__(self):
        self.saved_context = getcontext()
        setcontext(self.new_context)
        return self.new_context
    def __exit__(self, t, v, tb):
        setcontext(self.saved_context)

class Context(object):
    """Contains the context for a Decimal instance.

    Contains:
    prec - precision (for use in rounding, division, square roots..)
    rounding - rounding type (how you round)
    traps - If traps[exception] = 1, then the exception is
                    raised when it is caused.  Otherwise, a value is
                    substituted in.
    flags  - When an exception is caused, flags[exception] is incremented.
             (Whether or not the trap_enabler is set)
             Should be reset by user of Decimal instance.
    Emin -   Minimum exponent
    Emax -   Maximum exponent
    capitals -      If 1, 1*10^1 is printed as 1E+1.
                    If 0, printed as 1e1
    _clamp - If 1, change exponents if too high (Default 0)
    """

    def __init__(self, prec=None, rounding=None,
                 traps=None, flags=None,
                 Emin=None, Emax=None,
                 capitals=None, _clamp=0,
                 _ignored_flags=None):
        if flags is None:
            flags = []
        if _ignored_flags is None:
            _ignored_flags = []
        if not isinstance(flags, dict):
            flags = dict([(s,s in flags) for s in _signals])
            del s
        if traps is not None and not isinstance(traps, dict):
            traps = dict([(s,s in traps) for s in _signals])
            del s
        for name, val in locals().items():
            if val is None:
                setattr(self, name, _copy.copy(getattr(DefaultContext, name)))
            else:
                setattr(self, name, val)
        del self.self

    def __repr__(self):
        """Show the current context."""
        s = []
        s.append('Context(prec=%(prec)d, rounding=%(rounding)s, '
                 'Emin=%(Emin)d, Emax=%(Emax)d, capitals=%(capitals)d'
                 % vars(self))
        names = [f.__name__ for f, v in self.flags.items() if v]
        s.append('flags=[' + ', '.join(names) + ']')
        names = [t.__name__ for t, v in self.traps.items() if v]
        s.append('traps=[' + ', '.join(names) + ']')
        return ', '.join(s) + ')'

    def clear_flags(self):
        """Reset all flags to zero"""
        for flag in self.flags:
            self.flags[flag] = 0

    def _shallow_copy(self):
        """Returns a shallow copy from self."""
        nc = Context(self.prec, self.rounding, self.traps,
                     self.flags, self.Emin, self.Emax,
                     self.capitals, self._clamp, self._ignored_flags)
        return nc

    def copy(self):
        """Returns a deep copy from self."""
        nc = Context(self.prec, self.rounding, self.traps.copy(),
                     self.flags.copy(), self.Emin, self.Emax,
                     self.capitals, self._clamp, self._ignored_flags)
        return nc
    __copy__ = copy

    def _raise_error(self, condition, explanation = None, *args):
        """Handles an error

        If the flag is in _ignored_flags, returns the default response.
        Otherwise, it increments the flag, then, if the corresponding
        trap_enabler is set, it reaises the exception.  Otherwise, it returns
        the default value after incrementing the flag.
        """
        error = _condition_map.get(condition, condition)
        if error in self._ignored_flags:
            # Don't touch the flag
            return error().handle(self, *args)

        self.flags[error] += 1
        if not self.traps[error]:
            # The errors define how to handle themselves.
            return condition().handle(self, *args)

        # Errors should only be risked on copies of the context
        # self._ignored_flags = []
        raise error, explanation

    def _ignore_all_flags(self):
        """Ignore all flags, if they are raised"""
        return self._ignore_flags(*_signals)

    def _ignore_flags(self, *flags):
        """Ignore the flags, if they are raised"""
        # Do not mutate-- This way, copies of a context leave the original
        # alone.
        self._ignored_flags = (self._ignored_flags + list(flags))
        return list(flags)

    def _regard_flags(self, *flags):
        """Stop ignoring the flags, if they are raised"""
        if flags and isinstance(flags[0], (tuple,list)):
            flags = flags[0]
        for flag in flags:
            self._ignored_flags.remove(flag)

    def __hash__(self):
        """A Context cannot be hashed."""
        # We inherit object.__hash__, so we must deny this explicitly
        raise TypeError("Cannot hash a Context.")

    def Etiny(self):
        """Returns Etiny (= Emin - prec + 1)"""
        return int(self.Emin - self.prec + 1)

    def Etop(self):
        """Returns maximum exponent (= Emax - prec + 1)"""
        return int(self.Emax - self.prec + 1)

    def _set_rounding(self, type):
        """Sets the rounding type.

        Sets the rounding type, and returns the current (previous)
        rounding type.  Often used like:

        context = context.copy()
        # so you don't change the calling context
        # if an error occurs in the middle.
        rounding = context._set_rounding(ROUND_UP)
        val = self.__sub__(other, context=context)
        context._set_rounding(rounding)

        This will make it round up for that operation.
        """
        rounding = self.rounding
        self.rounding= type
        return rounding

    def create_decimal(self, num='0'):
        """Creates a new Decimal instance but using self as context."""
        d = Decimal(num, context=self)
        if d._isnan() and len(d._int) > self.prec - self._clamp:
            return self._raise_error(ConversionSyntax,
                                     "diagnostic info too long in NaN")
        return d._fix(self)

    # Methods
    def abs(self, a):
        """Returns the absolute value of the operand.

        If the operand is negative, the result is the same as using the minus
        operation on the operand.  Otherwise, the result is the same as using
        the plus operation on the operand.

        >>> ExtendedContext.abs(Decimal('2.1'))
        Decimal("2.1")
        >>> ExtendedContext.abs(Decimal('-100'))
        Decimal("100")
        >>> ExtendedContext.abs(Decimal('101.5'))
        Decimal("101.5")
        >>> ExtendedContext.abs(Decimal('-101.5'))
        Decimal("101.5")
        """
        return a.__abs__(context=self)

    def add(self, a, b):
        """Return the sum of the two operands.

        >>> ExtendedContext.add(Decimal('12'), Decimal('7.00'))
        Decimal("19.00")
        >>> ExtendedContext.add(Decimal('1E+2'), Decimal('1.01E+4'))
        Decimal("1.02E+4")
        """
        return a.__add__(b, context=self)

    def _apply(self, a):
        return str(a._fix(self))

    def canonical(self, a):
        """Returns the same Decimal object.

        As we do not have different encodings for the same number, the
        received object already is in its canonical form.

        >>> ExtendedContext.canonical(Decimal('2.50'))
        Decimal("2.50")
        """
        return a.canonical(context=self)

    def compare(self, a, b):
        """Compares values numerically.

        If the signs of the operands differ, a value representing each operand
        ('-1' if the operand is less than zero, '0' if the operand is zero or
        negative zero, or '1' if the operand is greater than zero) is used in
        place of that operand for the comparison instead of the actual
        operand.

        The comparison is then effected by subtracting the second operand from
        the first and then returning a value according to the result of the
        subtraction: '-1' if the result is less than zero, '0' if the result is
        zero or negative zero, or '1' if the result is greater than zero.

        >>> ExtendedContext.compare(Decimal('2.1'), Decimal('3'))
        Decimal("-1")
        >>> ExtendedContext.compare(Decimal('2.1'), Decimal('2.1'))
        Decimal("0")
        >>> ExtendedContext.compare(Decimal('2.1'), Decimal('2.10'))
        Decimal("0")
        >>> ExtendedContext.compare(Decimal('3'), Decimal('2.1'))
        Decimal("1")
        >>> ExtendedContext.compare(Decimal('2.1'), Decimal('-3'))
        Decimal("1")
        >>> ExtendedContext.compare(Decimal('-3'), Decimal('2.1'))
        Decimal("-1")
        """
        return a.compare(b, context=self)

    def compare_signal(self, a, b):
        """Compares the values of the two operands numerically.

        It's pretty much like compare(), but all NaNs signal, with signaling
        NaNs taking precedence over quiet NaNs.

        >>> c = ExtendedContext
        >>> c.compare_signal(Decimal('2.1'), Decimal('3'))
        Decimal("-1")
        >>> c.compare_signal(Decimal('2.1'), Decimal('2.1'))
        Decimal("0")
        >>> c.flags[InvalidOperation] = 0
        >>> print c.flags[InvalidOperation]
        0
        >>> c.compare_signal(Decimal('NaN'), Decimal('2.1'))
        Decimal("NaN")
        >>> print c.flags[InvalidOperation]
        1
        >>> c.flags[InvalidOperation] = 0
        >>> print c.flags[InvalidOperation]
        0
        >>> c.compare_signal(Decimal('sNaN'), Decimal('2.1'))
        Decimal("NaN")
        >>> print c.flags[InvalidOperation]
        1
        """
        return a.compare_signal(b, context=self)

    def compare_total(self, a, b):
        """Compares two operands using their abstract representation.

        This is not like the standard compare, which use their numerical
        value. Note that a total ordering is defined for all possible abstract
        representations.

        >>> ExtendedContext.compare_total(Decimal('12.73'), Decimal('127.9'))
        Decimal("-1")
        >>> ExtendedContext.compare_total(Decimal('-127'),  Decimal('12'))
        Decimal("-1")
        >>> ExtendedContext.compare_total(Decimal('12.30'), Decimal('12.3'))
        Decimal("-1")
        >>> ExtendedContext.compare_total(Decimal('12.30'), Decimal('12.30'))
        Decimal("0")
        >>> ExtendedContext.compare_total(Decimal('12.3'),  Decimal('12.300'))
        Decimal("1")
        >>> ExtendedContext.compare_total(Decimal('12.3'),  Decimal('NaN'))
        Decimal("-1")
        """
        return a.compare_total(b)

    def compare_total_mag(self, a, b):
        """Compares two operands using their abstract representation ignoring sign.

        Like compare_total, but with operand's sign ignored and assumed to be 0.
        """
        return a.compare_total_mag(b)

    def copy_abs(self, a):
        """Returns a copy of the operand with the sign set to 0.

        >>> ExtendedContext.copy_abs(Decimal('2.1'))
        Decimal("2.1")
        >>> ExtendedContext.copy_abs(Decimal('-100'))
        Decimal("100")
        """
        return a.copy_abs()

    def copy_decimal(self, a):
        """Returns a copy of the decimal objet.

        >>> ExtendedContext.copy_decimal(Decimal('2.1'))
        Decimal("2.1")
        >>> ExtendedContext.copy_decimal(Decimal('-1.00'))
        Decimal("-1.00")
        """
        return Decimal(a)

    def copy_negate(self, a):
        """Returns a copy of the operand with the sign inverted.

        >>> ExtendedContext.copy_negate(Decimal('101.5'))
        Decimal("-101.5")
        >>> ExtendedContext.copy_negate(Decimal('-101.5'))
        Decimal("101.5")
        """
        return a.copy_negate()

    def copy_sign(self, a, b):
        """Copies the second operand's sign to the first one.

        In detail, it returns a copy of the first operand with the sign
        equal to the sign of the second operand.

        >>> ExtendedContext.copy_sign(Decimal( '1.50'), Decimal('7.33'))
        Decimal("1.50")
        >>> ExtendedContext.copy_sign(Decimal('-1.50'), Decimal('7.33'))
        Decimal("1.50")
        >>> ExtendedContext.copy_sign(Decimal( '1.50'), Decimal('-7.33'))
        Decimal("-1.50")
        >>> ExtendedContext.copy_sign(Decimal('-1.50'), Decimal('-7.33'))
        Decimal("-1.50")
        """
        return a.copy_sign(b)

    def divide(self, a, b):
        """Decimal division in a specified context.

        >>> ExtendedContext.divide(Decimal('1'), Decimal('3'))
        Decimal("0.333333333")
        >>> ExtendedContext.divide(Decimal('2'), Decimal('3'))
        Decimal("0.666666667")
        >>> ExtendedContext.divide(Decimal('5'), Decimal('2'))
        Decimal("2.5")
        >>> ExtendedContext.divide(Decimal('1'), Decimal('10'))
        Decimal("0.1")
        >>> ExtendedContext.divide(Decimal('12'), Decimal('12'))
        Decimal("1")
        >>> ExtendedContext.divide(Decimal('8.00'), Decimal('2'))
        Decimal("4.00")
        >>> ExtendedContext.divide(Decimal('2.400'), Decimal('2.0'))
        Decimal("1.20")
        >>> ExtendedContext.divide(Decimal('1000'), Decimal('100'))
        Decimal("10")
        >>> ExtendedContext.divide(Decimal('1000'), Decimal('1'))
        Decimal("1000")
        >>> ExtendedContext.divide(Decimal('2.40E+6'), Decimal('2'))
        Decimal("1.20E+6")
        """
        return a.__div__(b, context=self)

    def divide_int(self, a, b):
        """Divides two numbers and returns the integer part of the result.

        >>> ExtendedContext.divide_int(Decimal('2'), Decimal('3'))
        Decimal("0")
        >>> ExtendedContext.divide_int(Decimal('10'), Decimal('3'))
        Decimal("3")
        >>> ExtendedContext.divide_int(Decimal('1'), Decimal('0.3'))
        Decimal("3")
        """
        return a.__floordiv__(b, context=self)

    def divmod(self, a, b):
        return a.__divmod__(b, context=self)

    def exp(self, a):
        """Returns e ** a.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.exp(Decimal('-Infinity'))
        Decimal("0")
        >>> c.exp(Decimal('-1'))
        Decimal("0.367879441")
        >>> c.exp(Decimal('0'))
        Decimal("1")
        >>> c.exp(Decimal('1'))
        Decimal("2.71828183")
        >>> c.exp(Decimal('0.693147181'))
        Decimal("2.00000000")
        >>> c.exp(Decimal('+Infinity'))
        Decimal("Infinity")
        """
        return a.exp(context=self)

    def fma(self, a, b, c):
        """Returns a multiplied by b, plus c.

        The first two operands are multiplied together, using multiply,
        the third operand is then added to the result of that
        multiplication, using add, all with only one final rounding.

        >>> ExtendedContext.fma(Decimal('3'), Decimal('5'), Decimal('7'))
        Decimal("22")
        >>> ExtendedContext.fma(Decimal('3'), Decimal('-5'), Decimal('7'))
        Decimal("-8")
        >>> ExtendedContext.fma(Decimal('888565290'), Decimal('1557.96930'), Decimal('-86087.7578'))
        Decimal("1.38435736E+12")
        """
        return a.fma(b, c, context=self)

    def is_canonical(self, a):
        """Return True if the operand is canonical; otherwise return False.

        Currently, the encoding of a Decimal instance is always
        canonical, so this method returns True for any Decimal.

        >>> ExtendedContext.is_canonical(Decimal('2.50'))
        True
        """
        return a.is_canonical()

    def is_finite(self, a):
        """Return True if the operand is finite; otherwise return False.

        A Decimal instance is considered finite if it is neither
        infinite nor a NaN.

        >>> ExtendedContext.is_finite(Decimal('2.50'))
        True
        >>> ExtendedContext.is_finite(Decimal('-0.3'))
        True
        >>> ExtendedContext.is_finite(Decimal('0'))
        True
        >>> ExtendedContext.is_finite(Decimal('Inf'))
        False
        >>> ExtendedContext.is_finite(Decimal('NaN'))
        False
        """
        return a.is_finite()

    def is_infinite(self, a):
        """Return True if the operand is infinite; otherwise return False.

        >>> ExtendedContext.is_infinite(Decimal('2.50'))
        False
        >>> ExtendedContext.is_infinite(Decimal('-Inf'))
        True
        >>> ExtendedContext.is_infinite(Decimal('NaN'))
        False
        """
        return a.is_infinite()

    def is_nan(self, a):
        """Return True if the operand is a qNaN or sNaN;
        otherwise return False.

        >>> ExtendedContext.is_nan(Decimal('2.50'))
        False
        >>> ExtendedContext.is_nan(Decimal('NaN'))
        True
        >>> ExtendedContext.is_nan(Decimal('-sNaN'))
        True
        """
        return a.is_nan()

    def is_normal(self, a):
        """Return True if the operand is a normal number;
        otherwise return False.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.is_normal(Decimal('2.50'))
        True
        >>> c.is_normal(Decimal('0.1E-999'))
        False
        >>> c.is_normal(Decimal('0.00'))
        False
        >>> c.is_normal(Decimal('-Inf'))
        False
        >>> c.is_normal(Decimal('NaN'))
        False
        """
        return a.is_normal(context=self)

    def is_qnan(self, a):
        """Return True if the operand is a quiet NaN; otherwise return False.

        >>> ExtendedContext.is_qnan(Decimal('2.50'))
        False
        >>> ExtendedContext.is_qnan(Decimal('NaN'))
        True
        >>> ExtendedContext.is_qnan(Decimal('sNaN'))
        False
        """
        return a.is_qnan()

    def is_signed(self, a):
        """Return True if the operand is negative; otherwise return False.

        >>> ExtendedContext.is_signed(Decimal('2.50'))
        False
        >>> ExtendedContext.is_signed(Decimal('-12'))
        True
        >>> ExtendedContext.is_signed(Decimal('-0'))
        True
        """
        return a.is_signed()

    def is_snan(self, a):
        """Return True if the operand is a signaling NaN;
        otherwise return False.

        >>> ExtendedContext.is_snan(Decimal('2.50'))
        False
        >>> ExtendedContext.is_snan(Decimal('NaN'))
        False
        >>> ExtendedContext.is_snan(Decimal('sNaN'))
        True
        """
        return a.is_snan()

    def is_subnormal(self, a):
        """Return True if the operand is subnormal; otherwise return False.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.is_subnormal(Decimal('2.50'))
        False
        >>> c.is_subnormal(Decimal('0.1E-999'))
        True
        >>> c.is_subnormal(Decimal('0.00'))
        False
        >>> c.is_subnormal(Decimal('-Inf'))
        False
        >>> c.is_subnormal(Decimal('NaN'))
        False
        """
        return a.is_subnormal(context=self)

    def is_zero(self, a):
        """Return True if the operand is a zero; otherwise return False.

        >>> ExtendedContext.is_zero(Decimal('0'))
        True
        >>> ExtendedContext.is_zero(Decimal('2.50'))
        False
        >>> ExtendedContext.is_zero(Decimal('-0E+2'))
        True
        """
        return a.is_zero()

    def ln(self, a):
        """Returns the natural (base e) logarithm of the operand.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.ln(Decimal('0'))
        Decimal("-Infinity")
        >>> c.ln(Decimal('1.000'))
        Decimal("0")
        >>> c.ln(Decimal('2.71828183'))
        Decimal("1.00000000")
        >>> c.ln(Decimal('10'))
        Decimal("2.30258509")
        >>> c.ln(Decimal('+Infinity'))
        Decimal("Infinity")
        """
        return a.ln(context=self)

    def log10(self, a):
        """Returns the base 10 logarithm of the operand.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.log10(Decimal('0'))
        Decimal("-Infinity")
        >>> c.log10(Decimal('0.001'))
        Decimal("-3")
        >>> c.log10(Decimal('1.000'))
        Decimal("0")
        >>> c.log10(Decimal('2'))
        Decimal("0.301029996")
        >>> c.log10(Decimal('10'))
        Decimal("1")
        >>> c.log10(Decimal('70'))
        Decimal("1.84509804")
        >>> c.log10(Decimal('+Infinity'))
        Decimal("Infinity")
        """
        return a.log10(context=self)

    def logb(self, a):
        """ Returns the exponent of the magnitude of the operand's MSD.

        The result is the integer which is the exponent of the magnitude
        of the most significant digit of the operand (as though the
        operand were truncated to a single digit while maintaining the
        value of that digit and without limiting the resulting exponent).

        >>> ExtendedContext.logb(Decimal('250'))
        Decimal("2")
        >>> ExtendedContext.logb(Decimal('2.50'))
        Decimal("0")
        >>> ExtendedContext.logb(Decimal('0.03'))
        Decimal("-2")
        >>> ExtendedContext.logb(Decimal('0'))
        Decimal("-Infinity")
        """
        return a.logb(context=self)

    def logical_and(self, a, b):
        """Applies the logical operation 'and' between each operand's digits.

        The operands must be both logical numbers.

        >>> ExtendedContext.logical_and(Decimal('0'), Decimal('0'))
        Decimal("0")
        >>> ExtendedContext.logical_and(Decimal('0'), Decimal('1'))
        Decimal("0")
        >>> ExtendedContext.logical_and(Decimal('1'), Decimal('0'))
        Decimal("0")
        >>> ExtendedContext.logical_and(Decimal('1'), Decimal('1'))
        Decimal("1")
        >>> ExtendedContext.logical_and(Decimal('1100'), Decimal('1010'))
        Decimal("1000")
        >>> ExtendedContext.logical_and(Decimal('1111'), Decimal('10'))
        Decimal("10")
        """
        return a.logical_and(b, context=self)

    def logical_invert(self, a):
        """Invert all the digits in the operand.

        The operand must be a logical number.

        >>> ExtendedContext.logical_invert(Decimal('0'))
        Decimal("111111111")
        >>> ExtendedContext.logical_invert(Decimal('1'))
        Decimal("111111110")
        >>> ExtendedContext.logical_invert(Decimal('111111111'))
        Decimal("0")
        >>> ExtendedContext.logical_invert(Decimal('101010101'))
        Decimal("10101010")
        """
        return a.logical_invert(context=self)

    def logical_or(self, a, b):
        """Applies the logical operation 'or' between each operand's digits.

        The operands must be both logical numbers.

        >>> ExtendedContext.logical_or(Decimal('0'), Decimal('0'))
        Decimal("0")
        >>> ExtendedContext.logical_or(Decimal('0'), Decimal('1'))
        Decimal("1")
        >>> ExtendedContext.logical_or(Decimal('1'), Decimal('0'))
        Decimal("1")
        >>> ExtendedContext.logical_or(Decimal('1'), Decimal('1'))
        Decimal("1")
        >>> ExtendedContext.logical_or(Decimal('1100'), Decimal('1010'))
        Decimal("1110")
        >>> ExtendedContext.logical_or(Decimal('1110'), Decimal('10'))
        Decimal("1110")
        """
        return a.logical_or(b, context=self)

    def logical_xor(self, a, b):
        """Applies the logical operation 'xor' between each operand's digits.

        The operands must be both logical numbers.

        >>> ExtendedContext.logical_xor(Decimal('0'), Decimal('0'))
        Decimal("0")
        >>> ExtendedContext.logical_xor(Decimal('0'), Decimal('1'))
        Decimal("1")
        >>> ExtendedContext.logical_xor(Decimal('1'), Decimal('0'))
        Decimal("1")
        >>> ExtendedContext.logical_xor(Decimal('1'), Decimal('1'))
        Decimal("0")
        >>> ExtendedContext.logical_xor(Decimal('1100'), Decimal('1010'))
        Decimal("110")
        >>> ExtendedContext.logical_xor(Decimal('1111'), Decimal('10'))
        Decimal("1101")
        """
        return a.logical_xor(b, context=self)

    def max(self, a,b):
        """max compares two values numerically and returns the maximum.

        If either operand is a NaN then the general rules apply.
        Otherwise, the operands are compared as as though by the compare
        operation.  If they are numerically equal then the left-hand operand
        is chosen as the result.  Otherwise the maximum (closer to positive
        infinity) of the two operands is chosen as the result.

        >>> ExtendedContext.max(Decimal('3'), Decimal('2'))
        Decimal("3")
        >>> ExtendedContext.max(Decimal('-10'), Decimal('3'))
        Decimal("3")
        >>> ExtendedContext.max(Decimal('1.0'), Decimal('1'))
        Decimal("1")
        >>> ExtendedContext.max(Decimal('7'), Decimal('NaN'))
        Decimal("7")
        """
        return a.max(b, context=self)

    def max_mag(self, a, b):
        """Compares the values numerically with their sign ignored."""
        return a.max_mag(b, context=self)

    def min(self, a,b):
        """min compares two values numerically and returns the minimum.

        If either operand is a NaN then the general rules apply.
        Otherwise, the operands are compared as as though by the compare
        operation.  If they are numerically equal then the left-hand operand
        is chosen as the result.  Otherwise the minimum (closer to negative
        infinity) of the two operands is chosen as the result.

        >>> ExtendedContext.min(Decimal('3'), Decimal('2'))
        Decimal("2")
        >>> ExtendedContext.min(Decimal('-10'), Decimal('3'))
        Decimal("-10")
        >>> ExtendedContext.min(Decimal('1.0'), Decimal('1'))
        Decimal("1.0")
        >>> ExtendedContext.min(Decimal('7'), Decimal('NaN'))
        Decimal("7")
        """
        return a.min(b, context=self)

    def min_mag(self, a, b):
        """Compares the values numerically with their sign ignored."""
        return a.min_mag(b, context=self)

    def minus(self, a):
        """Minus corresponds to unary prefix minus in Python.

        The operation is evaluated using the same rules as subtract; the
        operation minus(a) is calculated as subtract('0', a) where the '0'
        has the same exponent as the operand.

        >>> ExtendedContext.minus(Decimal('1.3'))
        Decimal("-1.3")
        >>> ExtendedContext.minus(Decimal('-1.3'))
        Decimal("1.3")
        """
        return a.__neg__(context=self)

    def multiply(self, a, b):
        """multiply multiplies two operands.

        If either operand is a special value then the general rules apply.
        Otherwise, the operands are multiplied together ('long multiplication'),
        resulting in a number which may be as long as the sum of the lengths
        of the two operands.

        >>> ExtendedContext.multiply(Decimal('1.20'), Decimal('3'))
        Decimal("3.60")
        >>> ExtendedContext.multiply(Decimal('7'), Decimal('3'))
        Decimal("21")
        >>> ExtendedContext.multiply(Decimal('0.9'), Decimal('0.8'))
        Decimal("0.72")
        >>> ExtendedContext.multiply(Decimal('0.9'), Decimal('-0'))
        Decimal("-0.0")
        >>> ExtendedContext.multiply(Decimal('654321'), Decimal('654321'))
        Decimal("4.28135971E+11")
        """
        return a.__mul__(b, context=self)

    def next_minus(self, a):
        """Returns the largest representable number smaller than a.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> ExtendedContext.next_minus(Decimal('1'))
        Decimal("0.999999999")
        >>> c.next_minus(Decimal('1E-1007'))
        Decimal("0E-1007")
        >>> ExtendedContext.next_minus(Decimal('-1.00000003'))
        Decimal("-1.00000004")
        >>> c.next_minus(Decimal('Infinity'))
        Decimal("9.99999999E+999")
        """
        return a.next_minus(context=self)

    def next_plus(self, a):
        """Returns the smallest representable number larger than a.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> ExtendedContext.next_plus(Decimal('1'))
        Decimal("1.00000001")
        >>> c.next_plus(Decimal('-1E-1007'))
        Decimal("-0E-1007")
        >>> ExtendedContext.next_plus(Decimal('-1.00000003'))
        Decimal("-1.00000002")
        >>> c.next_plus(Decimal('-Infinity'))
        Decimal("-9.99999999E+999")
        """
        return a.next_plus(context=self)

    def next_toward(self, a, b):
        """Returns the number closest to a, in direction towards b.

        The result is the closest representable number from the first
        operand (but not the first operand) that is in the direction
        towards the second operand, unless the operands have the same
        value.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.next_toward(Decimal('1'), Decimal('2'))
        Decimal("1.00000001")
        >>> c.next_toward(Decimal('-1E-1007'), Decimal('1'))
        Decimal("-0E-1007")
        >>> c.next_toward(Decimal('-1.00000003'), Decimal('0'))
        Decimal("-1.00000002")
        >>> c.next_toward(Decimal('1'), Decimal('0'))
        Decimal("0.999999999")
        >>> c.next_toward(Decimal('1E-1007'), Decimal('-100'))
        Decimal("0E-1007")
        >>> c.next_toward(Decimal('-1.00000003'), Decimal('-10'))
        Decimal("-1.00000004")
        >>> c.next_toward(Decimal('0.00'), Decimal('-0.0000'))
        Decimal("-0.00")
        """
        return a.next_toward(b, context=self)

    def normalize(self, a):
        """normalize reduces an operand to its simplest form.

        Essentially a plus operation with all trailing zeros removed from the
        result.

        >>> ExtendedContext.normalize(Decimal('2.1'))
        Decimal("2.1")
        >>> ExtendedContext.normalize(Decimal('-2.0'))
        Decimal("-2")
        >>> ExtendedContext.normalize(Decimal('1.200'))
        Decimal("1.2")
        >>> ExtendedContext.normalize(Decimal('-120'))
        Decimal("-1.2E+2")
        >>> ExtendedContext.normalize(Decimal('120.00'))
        Decimal("1.2E+2")
        >>> ExtendedContext.normalize(Decimal('0.00'))
        Decimal("0")
        """
        return a.normalize(context=self)

    def number_class(self, a):
        """Returns an indication of the class of the operand.

        The class is one of the following strings:
          -sNaN
          -NaN
          -Infinity
          -Normal
          -Subnormal
          -Zero
          +Zero
          +Subnormal
          +Normal
          +Infinity

        >>> c = Context(ExtendedContext)
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.number_class(Decimal('Infinity'))
        '+Infinity'
        >>> c.number_class(Decimal('1E-10'))
        '+Normal'
        >>> c.number_class(Decimal('2.50'))
        '+Normal'
        >>> c.number_class(Decimal('0.1E-999'))
        '+Subnormal'
        >>> c.number_class(Decimal('0'))
        '+Zero'
        >>> c.number_class(Decimal('-0'))
        '-Zero'
        >>> c.number_class(Decimal('-0.1E-999'))
        '-Subnormal'
        >>> c.number_class(Decimal('-1E-10'))
        '-Normal'
        >>> c.number_class(Decimal('-2.50'))
        '-Normal'
        >>> c.number_class(Decimal('-Infinity'))
        '-Infinity'
        >>> c.number_class(Decimal('NaN'))
        'NaN'
        >>> c.number_class(Decimal('-NaN'))
        'NaN'
        >>> c.number_class(Decimal('sNaN'))
        'sNaN'
        """
        return a.number_class(context=self)

    def plus(self, a):
        """Plus corresponds to unary prefix plus in Python.

        The operation is evaluated using the same rules as add; the
        operation plus(a) is calculated as add('0', a) where the '0'
        has the same exponent as the operand.

        >>> ExtendedContext.plus(Decimal('1.3'))
        Decimal("1.3")
        >>> ExtendedContext.plus(Decimal('-1.3'))
        Decimal("-1.3")
        """
        return a.__pos__(context=self)

    def power(self, a, b, modulo=None):
        """Raises a to the power of b, to modulo if given.

        With two arguments, compute a**b.  If a is negative then b
        must be integral.  The result will be inexact unless b is
        integral and the result is finite and can be expressed exactly
        in 'precision' digits.

        With three arguments, compute (a**b) % modulo.  For the
        three argument form, the following restrictions on the
        arguments hold:

         - all three arguments must be integral
         - b must be nonnegative
         - at least one of a or b must be nonzero
         - modulo must be nonzero and have at most 'precision' digits

        The result of pow(a, b, modulo) is identical to the result
        that would be obtained by computing (a**b) % modulo with
        unbounded precision, but is computed more efficiently.  It is
        always exact.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.power(Decimal('2'), Decimal('3'))
        Decimal("8")
        >>> c.power(Decimal('-2'), Decimal('3'))
        Decimal("-8")
        >>> c.power(Decimal('2'), Decimal('-3'))
        Decimal("0.125")
        >>> c.power(Decimal('1.7'), Decimal('8'))
        Decimal("69.7575744")
        >>> c.power(Decimal('10'), Decimal('0.301029996'))
        Decimal("2.00000000")
        >>> c.power(Decimal('Infinity'), Decimal('-1'))
        Decimal("0")
        >>> c.power(Decimal('Infinity'), Decimal('0'))
        Decimal("1")
        >>> c.power(Decimal('Infinity'), Decimal('1'))
        Decimal("Infinity")
        >>> c.power(Decimal('-Infinity'), Decimal('-1'))
        Decimal("-0")
        >>> c.power(Decimal('-Infinity'), Decimal('0'))
        Decimal("1")
        >>> c.power(Decimal('-Infinity'), Decimal('1'))
        Decimal("-Infinity")
        >>> c.power(Decimal('-Infinity'), Decimal('2'))
        Decimal("Infinity")
        >>> c.power(Decimal('0'), Decimal('0'))
        Decimal("NaN")

        >>> c.power(Decimal('3'), Decimal('7'), Decimal('16'))
        Decimal("11")
        >>> c.power(Decimal('-3'), Decimal('7'), Decimal('16'))
        Decimal("-11")
        >>> c.power(Decimal('-3'), Decimal('8'), Decimal('16'))
        Decimal("1")
        >>> c.power(Decimal('3'), Decimal('7'), Decimal('-16'))
        Decimal("11")
        >>> c.power(Decimal('23E12345'), Decimal('67E189'), Decimal('123456789'))
        Decimal("11729830")
        >>> c.power(Decimal('-0'), Decimal('17'), Decimal('1729'))
        Decimal("-0")
        >>> c.power(Decimal('-23'), Decimal('0'), Decimal('65537'))
        Decimal("1")
        """
        return a.__pow__(b, modulo, context=self)

    def quantize(self, a, b):
        """Returns a value equal to 'a' (rounded), having the exponent of 'b'.

        The coefficient of the result is derived from that of the left-hand
        operand.  It may be rounded using the current rounding setting (if the
        exponent is being increased), multiplied by a positive power of ten (if
        the exponent is being decreased), or is unchanged (if the exponent is
        already equal to that of the right-hand operand).

        Unlike other operations, if the length of the coefficient after the
        quantize operation would be greater than precision then an Invalid
        operation condition is raised.  This guarantees that, unless there is
        an error condition, the exponent of the result of a quantize is always
        equal to that of the right-hand operand.

        Also unlike other operations, quantize will never raise Underflow, even
        if the result is subnormal and inexact.

        >>> ExtendedContext.quantize(Decimal('2.17'), Decimal('0.001'))
        Decimal("2.170")
        >>> ExtendedContext.quantize(Decimal('2.17'), Decimal('0.01'))
        Decimal("2.17")
        >>> ExtendedContext.quantize(Decimal('2.17'), Decimal('0.1'))
        Decimal("2.2")
        >>> ExtendedContext.quantize(Decimal('2.17'), Decimal('1e+0'))
        Decimal("2")
        >>> ExtendedContext.quantize(Decimal('2.17'), Decimal('1e+1'))
        Decimal("0E+1")
        >>> ExtendedContext.quantize(Decimal('-Inf'), Decimal('Infinity'))
        Decimal("-Infinity")
        >>> ExtendedContext.quantize(Decimal('2'), Decimal('Infinity'))
        Decimal("NaN")
        >>> ExtendedContext.quantize(Decimal('-0.1'), Decimal('1'))
        Decimal("-0")
        >>> ExtendedContext.quantize(Decimal('-0'), Decimal('1e+5'))
        Decimal("-0E+5")
        >>> ExtendedContext.quantize(Decimal('+35236450.6'), Decimal('1e-2'))
        Decimal("NaN")
        >>> ExtendedContext.quantize(Decimal('-35236450.6'), Decimal('1e-2'))
        Decimal("NaN")
        >>> ExtendedContext.quantize(Decimal('217'), Decimal('1e-1'))
        Decimal("217.0")
        >>> ExtendedContext.quantize(Decimal('217'), Decimal('1e-0'))
        Decimal("217")
        >>> ExtendedContext.quantize(Decimal('217'), Decimal('1e+1'))
        Decimal("2.2E+2")
        >>> ExtendedContext.quantize(Decimal('217'), Decimal('1e+2'))
        Decimal("2E+2")
        """
        return a.quantize(b, context=self)

    def radix(self):
        """Just returns 10, as this is Decimal, :)

        >>> ExtendedContext.radix()
        Decimal("10")
        """
        return Decimal(10)

    def remainder(self, a, b):
        """Returns the remainder from integer division.

        The result is the residue of the dividend after the operation of
        calculating integer division as described for divide-integer, rounded
        to precision digits if necessary.  The sign of the result, if
        non-zero, is the same as that of the original dividend.

        This operation will fail under the same conditions as integer division
        (that is, if integer division on the same two operands would fail, the
        remainder cannot be calculated).

        >>> ExtendedContext.remainder(Decimal('2.1'), Decimal('3'))
        Decimal("2.1")
        >>> ExtendedContext.remainder(Decimal('10'), Decimal('3'))
        Decimal("1")
        >>> ExtendedContext.remainder(Decimal('-10'), Decimal('3'))
        Decimal("-1")
        >>> ExtendedContext.remainder(Decimal('10.2'), Decimal('1'))
        Decimal("0.2")
        >>> ExtendedContext.remainder(Decimal('10'), Decimal('0.3'))
        Decimal("0.1")
        >>> ExtendedContext.remainder(Decimal('3.6'), Decimal('1.3'))
        Decimal("1.0")
        """
        return a.__mod__(b, context=self)

    def remainder_near(self, a, b):
        """Returns to be "a - b * n", where n is the integer nearest the exact
        value of "x / b" (if two integers are equally near then the even one
        is chosen).  If the result is equal to 0 then its sign will be the
        sign of a.

        This operation will fail under the same conditions as integer division
        (that is, if integer division on the same two operands would fail, the
        remainder cannot be calculated).

        >>> ExtendedContext.remainder_near(Decimal('2.1'), Decimal('3'))
        Decimal("-0.9")
        >>> ExtendedContext.remainder_near(Decimal('10'), Decimal('6'))
        Decimal("-2")
        >>> ExtendedContext.remainder_near(Decimal('10'), Decimal('3'))
        Decimal("1")
        >>> ExtendedContext.remainder_near(Decimal('-10'), Decimal('3'))
        Decimal("-1")
        >>> ExtendedContext.remainder_near(Decimal('10.2'), Decimal('1'))
        Decimal("0.2")
        >>> ExtendedContext.remainder_near(Decimal('10'), Decimal('0.3'))
        Decimal("0.1")
        >>> ExtendedContext.remainder_near(Decimal('3.6'), Decimal('1.3'))
        Decimal("-0.3")
        """
        return a.remainder_near(b, context=self)

    def rotate(self, a, b):
        """Returns a rotated copy of a, b times.

        The coefficient of the result is a rotated copy of the digits in
        the coefficient of the first operand.  The number of places of
        rotation is taken from the absolute value of the second operand,
        with the rotation being to the left if the second operand is
        positive or to the right otherwise.

        >>> ExtendedContext.rotate(Decimal('34'), Decimal('8'))
        Decimal("400000003")
        >>> ExtendedContext.rotate(Decimal('12'), Decimal('9'))
        Decimal("12")
        >>> ExtendedContext.rotate(Decimal('123456789'), Decimal('-2'))
        Decimal("891234567")
        >>> ExtendedContext.rotate(Decimal('123456789'), Decimal('0'))
        Decimal("123456789")
        >>> ExtendedContext.rotate(Decimal('123456789'), Decimal('+2'))
        Decimal("345678912")
        """
        return a.rotate(b, context=self)

    def same_quantum(self, a, b):
        """Returns True if the two operands have the same exponent.

        The result is never affected by either the sign or the coefficient of
        either operand.

        >>> ExtendedContext.same_quantum(Decimal('2.17'), Decimal('0.001'))
        False
        >>> ExtendedContext.same_quantum(Decimal('2.17'), Decimal('0.01'))
        True
        >>> ExtendedContext.same_quantum(Decimal('2.17'), Decimal('1'))
        False
        >>> ExtendedContext.same_quantum(Decimal('Inf'), Decimal('-Inf'))
        True
        """
        return a.same_quantum(b)

    def scaleb (self, a, b):
        """Returns the first operand after adding the second value its exp.

        >>> ExtendedContext.scaleb(Decimal('7.50'), Decimal('-2'))
        Decimal("0.0750")
        >>> ExtendedContext.scaleb(Decimal('7.50'), Decimal('0'))
        Decimal("7.50")
        >>> ExtendedContext.scaleb(Decimal('7.50'), Decimal('3'))
        Decimal("7.50E+3")
        """
        return a.scaleb (b, context=self)

    def shift(self, a, b):
        """Returns a shifted copy of a, b times.

        The coefficient of the result is a shifted copy of the digits
        in the coefficient of the first operand.  The number of places
        to shift is taken from the absolute value of the second operand,
        with the shift being to the left if the second operand is
        positive or to the right otherwise.  Digits shifted into the
        coefficient are zeros.

        >>> ExtendedContext.shift(Decimal('34'), Decimal('8'))
        Decimal("400000000")
        >>> ExtendedContext.shift(Decimal('12'), Decimal('9'))
        Decimal("0")
        >>> ExtendedContext.shift(Decimal('123456789'), Decimal('-2'))
        Decimal("1234567")
        >>> ExtendedContext.shift(Decimal('123456789'), Decimal('0'))
        Decimal("123456789")
        >>> ExtendedContext.shift(Decimal('123456789'), Decimal('+2'))
        Decimal("345678900")
        """
        return a.shift(b, context=self)

    def sqrt(self, a):
        """Square root of a non-negative number to context precision.

        If the result must be inexact, it is rounded using the round-half-even
        algorithm.

        >>> ExtendedContext.sqrt(Decimal('0'))
        Decimal("0")
        >>> ExtendedContext.sqrt(Decimal('-0'))
        Decimal("-0")
        >>> ExtendedContext.sqrt(Decimal('0.39'))
        Decimal("0.624499800")
        >>> ExtendedContext.sqrt(Decimal('100'))
        Decimal("10")
        >>> ExtendedContext.sqrt(Decimal('1'))
        Decimal("1")
        >>> ExtendedContext.sqrt(Decimal('1.0'))
        Decimal("1.0")
        >>> ExtendedContext.sqrt(Decimal('1.00'))
        Decimal("1.0")
        >>> ExtendedContext.sqrt(Decimal('7'))
        Decimal("2.64575131")
        >>> ExtendedContext.sqrt(Decimal('10'))
        Decimal("3.16227766")
        >>> ExtendedContext.prec
        9
        """
        return a.sqrt(context=self)

    def subtract(self, a, b):
        """Return the difference between the two operands.

        >>> ExtendedContext.subtract(Decimal('1.3'), Decimal('1.07'))
        Decimal("0.23")
        >>> ExtendedContext.subtract(Decimal('1.3'), Decimal('1.30'))
        Decimal("0.00")
        >>> ExtendedContext.subtract(Decimal('1.3'), Decimal('2.07'))
        Decimal("-0.77")
        """
        return a.__sub__(b, context=self)

    def to_eng_string(self, a):
        """Converts a number to a string, using scientific notation.

        The operation is not affected by the context.
        """
        return a.to_eng_string(context=self)

    def to_sci_string(self, a):
        """Converts a number to a string, using scientific notation.

        The operation is not affected by the context.
        """
        return a.__str__(context=self)

    def to_integral_exact(self, a):
        """Rounds to an integer.

        When the operand has a negative exponent, the result is the same
        as using the quantize() operation using the given operand as the
        left-hand-operand, 1E+0 as the right-hand-operand, and the precision
        of the operand as the precision setting; Inexact and Rounded flags
        are allowed in this operation.  The rounding mode is taken from the
        context.

        >>> ExtendedContext.to_integral_exact(Decimal('2.1'))
        Decimal("2")
        >>> ExtendedContext.to_integral_exact(Decimal('100'))
        Decimal("100")
        >>> ExtendedContext.to_integral_exact(Decimal('100.0'))
        Decimal("100")
        >>> ExtendedContext.to_integral_exact(Decimal('101.5'))
        Decimal("102")
        >>> ExtendedContext.to_integral_exact(Decimal('-101.5'))
        Decimal("-102")
        >>> ExtendedContext.to_integral_exact(Decimal('10E+5'))
        Decimal("1.0E+6")
        >>> ExtendedContext.to_integral_exact(Decimal('7.89E+77'))
        Decimal("7.89E+77")
        >>> ExtendedContext.to_integral_exact(Decimal('-Inf'))
        Decimal("-Infinity")
        """
        return a.to_integral_exact(context=self)

    def to_integral_value(self, a):
        """Rounds to an integer.

        When the operand has a negative exponent, the result is the same
        as using the quantize() operation using the given operand as the
        left-hand-operand, 1E+0 as the right-hand-operand, and the precision
        of the operand as the precision setting, except that no flags will
        be set.  The rounding mode is taken from the context.

        >>> ExtendedContext.to_integral_value(Decimal('2.1'))
        Decimal("2")
        >>> ExtendedContext.to_integral_value(Decimal('100'))
        Decimal("100")
        >>> ExtendedContext.to_integral_value(Decimal('100.0'))
        Decimal("100")
        >>> ExtendedContext.to_integral_value(Decimal('101.5'))
        Decimal("102")
        >>> ExtendedContext.to_integral_value(Decimal('-101.5'))
        Decimal("-102")
        >>> ExtendedContext.to_integral_value(Decimal('10E+5'))
        Decimal("1.0E+6")
        >>> ExtendedContext.to_integral_value(Decimal('7.89E+77'))
        Decimal("7.89E+77")
        >>> ExtendedContext.to_integral_value(Decimal('-Inf'))
        Decimal("-Infinity")
        """
        return a.to_integral_value(context=self)

    # the method name changed, but we provide also the old one, for compatibility
    to_integral = to_integral_value

class _WorkRep(object):
    __slots__ = ('sign','int','exp')
    # sign: 0 or 1
    # int:  int or long
    # exp:  None, int, or string

    def __init__(self, value=None):
        if value is None:
            self.sign = None
            self.int = 0
            self.exp = None
        elif isinstance(value, Decimal):
            self.sign = value._sign
            self.int = int(value._int)
            self.exp = value._exp
        else:
            # assert isinstance(value, tuple)
            self.sign = value[0]
            self.int = value[1]
            self.exp = value[2]

    def __repr__(self):
        return "(%r, %r, %r)" % (self.sign, self.int, self.exp)

    __str__ = __repr__



def _normalize(op1, op2, prec = 0):
    """Normalizes op1, op2 to have the same exp and length of coefficient.

    Done during addition.
    """
    if op1.exp < op2.exp:
        tmp = op2
        other = op1
    else:
        tmp = op1
        other = op2

    # Let exp = min(tmp.exp - 1, tmp.adjusted() - precision - 1).
    # Then adding 10**exp to tmp has the same effect (after rounding)
    # as adding any positive quantity smaller than 10**exp; similarly
    # for subtraction.  So if other is smaller than 10**exp we replace
    # it with 10**exp.  This avoids tmp.exp - other.exp getting too large.
    tmp_len = len(str(tmp.int))
    other_len = len(str(other.int))
    exp = tmp.exp + min(-1, tmp_len - prec - 2)
    if other_len + other.exp - 1 < exp:
        other.int = 1
        other.exp = exp

    tmp.int *= 10 ** (tmp.exp - other.exp)
    tmp.exp = other.exp
    return op1, op2

##### Integer arithmetic functions used by ln, log10, exp and __pow__ #####

# This function from Tim Peters was taken from here:
# http://mail.python.org/pipermail/python-list/1999-July/007758.html
# The correction being in the function definition is for speed, and
# the whole function is not resolved with math.log because of avoiding
# the use of floats.
def _nbits(n, correction = {
        '0': 4, '1': 3, '2': 2, '3': 2,
        '4': 1, '5': 1, '6': 1, '7': 1,
        '8': 0, '9': 0, 'a': 0, 'b': 0,
        'c': 0, 'd': 0, 'e': 0, 'f': 0}):
    """Number of bits in binary representation of the positive integer n,
    or 0 if n == 0.
    """
    if n < 0:
        raise ValueError("The argument to _nbits should be nonnegative.")
    hex_n = "%x" % n
    return 4*len(hex_n) - correction[hex_n[0]]

def _sqrt_nearest(n, a):
    """Closest integer to the square root of the positive integer n.  a is
    an initial approximation to the square root.  Any positive integer
    will do for a, but the closer a is to the square root of n the
    faster convergence will be.

    """
    if n <= 0 or a <= 0:
        raise ValueError("Both arguments to _sqrt_nearest should be positive.")

    b=0
    while a != b:
        b, a = a, a--n//a>>1
    return a

def _rshift_nearest(x, shift):
    """Given an integer x and a nonnegative integer shift, return closest
    integer to x / 2**shift; use round-to-even in case of a tie.

    """
    b, q = 1L << shift, x >> shift
    return q + (2*(x & (b-1)) + (q&1) > b)

def _div_nearest(a, b):
    """Closest integer to a/b, a and b positive integers; rounds to even
    in the case of a tie.

    """
    q, r = divmod(a, b)
    return q + (2*r + (q&1) > b)

def _ilog(x, M, L = 8):
    """Integer approximation to M*log(x/M), with absolute error boundable
    in terms only of x/M.

    Given positive integers x and M, return an integer approximation to
    M * log(x/M).  For L = 8 and 0.1 <= x/M <= 10 the difference
    between the approximation and the exact result is at most 22.  For
    L = 8 and 1.0 <= x/M <= 10.0 the difference is at most 15.  In
    both cases these are upper bounds on the error; it will usually be
    much smaller."""

    # The basic algorithm is the following: let log1p be the function
    # log1p(x) = log(1+x).  Then log(x/M) = log1p((x-M)/M).  We use
    # the reduction
    #
    #    log1p(y) = 2*log1p(y/(1+sqrt(1+y)))
    #
    # repeatedly until the argument to log1p is small (< 2**-L in
    # absolute value).  For small y we can use the Taylor series
    # expansion
    #
    #    log1p(y) ~ y - y**2/2 + y**3/3 - ... - (-y)**T/T
    #
    # truncating at T such that y**T is small enough.  The whole
    # computation is carried out in a form of fixed-point arithmetic,
    # with a real number z being represented by an integer
    # approximation to z*M.  To avoid loss of precision, the y below
    # is actually an integer approximation to 2**R*y*M, where R is the
    # number of reductions performed so far.

    y = x-M
    # argument reduction; R = number of reductions performed
    R = 0
    while (R <= L and long(abs(y)) << L-R >= M or
           R > L and abs(y) >> R-L >= M):
        y = _div_nearest(long(M*y) << 1,
                         M + _sqrt_nearest(M*(M+_rshift_nearest(y, R)), M))
        R += 1

    # Taylor series with T terms
    T = -int(-10*len(str(M))//(3*L))
    yshift = _rshift_nearest(y, R)
    w = _div_nearest(M, T)
    for k in xrange(T-1, 0, -1):
        w = _div_nearest(M, k) - _div_nearest(yshift*w, M)

    return _div_nearest(w*y, M)

def _dlog10(c, e, p):
    """Given integers c, e and p with c > 0, p >= 0, compute an integer
    approximation to 10**p * log10(c*10**e), with an absolute error of
    at most 1.  Assumes that c*10**e is not exactly 1."""

    # increase precision by 2; compensate for this by dividing
    # final result by 100
    p += 2

    # write c*10**e as d*10**f with either:
    #   f >= 0 and 1 <= d <= 10, or
    #   f <= 0 and 0.1 <= d <= 1.
    # Thus for c*10**e close to 1, f = 0
    l = len(str(c))
    f = e+l - (e+l >= 1)

    if p > 0:
        M = 10**p
        k = e+p-f
        if k >= 0:
            c *= 10**k
        else:
            c = _div_nearest(c, 10**-k)

        log_d = _ilog(c, M) # error < 5 + 22 = 27
        log_10 = _log10_digits(p) # error < 1
        log_d = _div_nearest(log_d*M, log_10)
        log_tenpower = f*M # exact
    else:
        log_d = 0  # error < 2.31
        log_tenpower = div_nearest(f, 10**-p) # error < 0.5

    return _div_nearest(log_tenpower+log_d, 100)

def _dlog(c, e, p):
    """Given integers c, e and p with c > 0, compute an integer
    approximation to 10**p * log(c*10**e), with an absolute error of
    at most 1.  Assumes that c*10**e is not exactly 1."""

    # Increase precision by 2. The precision increase is compensated
    # for at the end with a division by 100.
    p += 2

    # rewrite c*10**e as d*10**f with either f >= 0 and 1 <= d <= 10,
    # or f <= 0 and 0.1 <= d <= 1.  Then we can compute 10**p * log(c*10**e)
    # as 10**p * log(d) + 10**p*f * log(10).
    l = len(str(c))
    f = e+l - (e+l >= 1)

    # compute approximation to 10**p*log(d), with error < 27
    if p > 0:
        k = e+p-f
        if k >= 0:
            c *= 10**k
        else:
            c = _div_nearest(c, 10**-k)  # error of <= 0.5 in c

        # _ilog magnifies existing error in c by a factor of at most 10
        log_d = _ilog(c, 10**p) # error < 5 + 22 = 27
    else:
        # p <= 0: just approximate the whole thing by 0; error < 2.31
        log_d = 0

    # compute approximation to f*10**p*log(10), with error < 11.
    if f:
        extra = len(str(abs(f)))-1
        if p + extra >= 0:
            # error in f * _log10_digits(p+extra) < |f| * 1 = |f|
            # after division, error < |f|/10**extra + 0.5 < 10 + 0.5 < 11
            f_log_ten = _div_nearest(f*_log10_digits(p+extra), 10**extra)
        else:
            f_log_ten = 0
    else:
        f_log_ten = 0

    # error in sum < 11+27 = 38; error after division < 0.38 + 0.5 < 1
    return _div_nearest(f_log_ten + log_d, 100)

class _Log10Memoize(object):
    """Class to compute, store, and allow retrieval of, digits of the
    constant log(10) = 2.302585....  This constant is needed by
    Decimal.ln, Decimal.log10, Decimal.exp and Decimal.__pow__."""
    def __init__(self):
        self.digits = "23025850929940456840179914546843642076011014886"

    def getdigits(self, p):
        """Given an integer p >= 0, return floor(10**p)*log(10).

        For example, self.getdigits(3) returns 2302.
        """
        # digits are stored as a string, for quick conversion to
        # integer in the case that we've already computed enough
        # digits; the stored digits should always be correct
        # (truncated, not rounded to nearest).
        if p < 0:
            raise ValueError("p should be nonnegative")

        if p >= len(self.digits):
            # compute p+3, p+6, p+9, ... digits; continue until at
            # least one of the extra digits is nonzero
            extra = 3
            while True:
                # compute p+extra digits, correct to within 1ulp
                M = 10**(p+extra+2)
                digits = str(_div_nearest(_ilog(10*M, M), 100))
                if digits[-extra:] != '0'*extra:
                    break
                extra += 3
            # keep all reliable digits so far; remove trailing zeros
            # and next nonzero digit
            self.digits = digits.rstrip('0')[:-1]
        return int(self.digits[:p+1])

_log10_digits = _Log10Memoize().getdigits

def _iexp(x, M, L=8):
    """Given integers x and M, M > 0, such that x/M is small in absolute
    value, compute an integer approximation to M*exp(x/M).  For 0 <=
    x/M <= 2.4, the absolute error in the result is bounded by 60 (and
    is usually much smaller)."""

    # Algorithm: to compute exp(z) for a real number z, first divide z
    # by a suitable power R of 2 so that |z/2**R| < 2**-L.  Then
    # compute expm1(z/2**R) = exp(z/2**R) - 1 using the usual Taylor
    # series
    #
    #     expm1(x) = x + x**2/2! + x**3/3! + ...
    #
    # Now use the identity
    #
    #     expm1(2x) = expm1(x)*(expm1(x)+2)
    #
    # R times to compute the sequence expm1(z/2**R),
    # expm1(z/2**(R-1)), ... , exp(z/2), exp(z).

    # Find R such that x/2**R/M <= 2**-L
    R = _nbits((long(x)<<L)//M)

    # Taylor series.  (2**L)**T > M
    T = -int(-10*len(str(M))//(3*L))
    y = _div_nearest(x, T)
    Mshift = long(M)<<R
    for i in xrange(T-1, 0, -1):
        y = _div_nearest(x*(Mshift + y), Mshift * i)

    # Expansion
    for k in xrange(R-1, -1, -1):
        Mshift = long(M)<<(k+2)
        y = _div_nearest(y*(y+Mshift), Mshift)

    return M+y

def _dexp(c, e, p):
    """Compute an approximation to exp(c*10**e), with p decimal places of
    precision.

    Returns integers d, f such that:

      10**(p-1) <= d <= 10**p, and
      (d-1)*10**f < exp(c*10**e) < (d+1)*10**f

    In other words, d*10**f is an approximation to exp(c*10**e) with p
    digits of precision, and with an error in d of at most 1.  This is
    almost, but not quite, the same as the error being < 1ulp: when d
    = 10**(p-1) the error could be up to 10 ulp."""

    # we'll call iexp with M = 10**(p+2), giving p+3 digits of precision
    p += 2

    # compute log(10) with extra precision = adjusted exponent of c*10**e
    extra = max(0, e + len(str(c)) - 1)
    q = p + extra

    # compute quotient c*10**e/(log(10)) = c*10**(e+q)/(log(10)*10**q),
    # rounding down
    shift = e+q
    if shift >= 0:
        cshift = c*10**shift
    else:
        cshift = c//10**-shift
    quot, rem = divmod(cshift, _log10_digits(q))

    # reduce remainder back to original precision
    rem = _div_nearest(rem, 10**extra)

    # error in result of _iexp < 120;  error after division < 0.62
    return _div_nearest(_iexp(rem, 10**p), 1000), quot - p + 3

def _dpower(xc, xe, yc, ye, p):
    """Given integers xc, xe, yc and ye representing Decimals x = xc*10**xe and
    y = yc*10**ye, compute x**y.  Returns a pair of integers (c, e) such that:

      10**(p-1) <= c <= 10**p, and
      (c-1)*10**e < x**y < (c+1)*10**e

    in other words, c*10**e is an approximation to x**y with p digits
    of precision, and with an error in c of at most 1.  (This is
    almost, but not quite, the same as the error being < 1ulp: when c
    == 10**(p-1) we can only guarantee error < 10ulp.)

    We assume that: x is positive and not equal to 1, and y is nonzero.
    """

    # Find b such that 10**(b-1) <= |y| <= 10**b
    b = len(str(abs(yc))) + ye

    # log(x) = lxc*10**(-p-b-1), to p+b+1 places after the decimal point
    lxc = _dlog(xc, xe, p+b+1)

    # compute product y*log(x) = yc*lxc*10**(-p-b-1+ye) = pc*10**(-p-1)
    shift = ye-b
    if shift >= 0:
        pc = lxc*yc*10**shift
    else:
        pc = _div_nearest(lxc*yc, 10**-shift)

    if pc == 0:
        # we prefer a result that isn't exactly 1; this makes it
        # easier to compute a correctly rounded result in __pow__
        if ((len(str(xc)) + xe >= 1) == (yc > 0)): # if x**y > 1:
            coeff, exp = 10**(p-1)+1, 1-p
        else:
            coeff, exp = 10**p-1, -p
    else:
        coeff, exp = _dexp(pc, -(p+1), p+1)
        coeff = _div_nearest(coeff, 10)
        exp += 1

    return coeff, exp

def _log10_lb(c, correction = {
        '1': 100, '2': 70, '3': 53, '4': 40, '5': 31,
        '6': 23, '7': 16, '8': 10, '9': 5}):
    """Compute a lower bound for 100*log10(c) for a positive integer c."""
    if c <= 0:
        raise ValueError("The argument to _log10_lb should be nonnegative.")
    str_c = str(c)
    return 100*len(str_c) - correction[str_c[0]]

##### Helper Functions ####################################################

def _convert_other(other, raiseit=False):
    """Convert other to Decimal.

    Verifies that it's ok to use in an implicit construction.
    """
    if isinstance(other, Decimal):
        return other
    if isinstance(other, (int, long)):
        return Decimal(other)
    if raiseit:
        raise TypeError("Unable to convert %s to Decimal" % other)
    return NotImplemented

##### Setup Specific Contexts ############################################

# The default context prototype used by Context()
# Is mutable, so that new contexts can have different default values

DefaultContext = Context(
        prec=28, rounding=ROUND_HALF_EVEN,
        traps=[DivisionByZero, Overflow, InvalidOperation],
        flags=[],
        Emax=999999999,
        Emin=-999999999,
        capitals=1
)

# Pre-made alternate contexts offered by the specification
# Don't change these; the user should be able to select these
# contexts and be able to reproduce results from other implementations
# of the spec.

BasicContext = Context(
        prec=9, rounding=ROUND_HALF_UP,
        traps=[DivisionByZero, Overflow, InvalidOperation, Clamped, Underflow],
        flags=[],
)

ExtendedContext = Context(
        prec=9, rounding=ROUND_HALF_EVEN,
        traps=[],
        flags=[],
)


##### crud for parsing strings #############################################
import re

# Regular expression used for parsing numeric strings.  Additional
# comments:
#
# 1. Uncomment the two '\s*' lines to allow leading and/or trailing
# whitespace.  But note that the specification disallows whitespace in
# a numeric string.
#
# 2. For finite numbers (not infinities and NaNs) the body of the
# number between the optional sign and the optional exponent must have
# at least one decimal digit, possibly after the decimal point.  The
# lookahead expression '(?=\d|\.\d)' checks this.
#
# As the flag UNICODE is not enabled here, we're explicitly avoiding any
# other meaning for \d than the numbers [0-9].

import re
_parser = re.compile(r"""     # A numeric string consists of:
#    \s*
    (?P<sign>[-+])?           # an optional sign, followed by either...
    (
        (?=\d|\.\d)           # ...a number (with at least one digit)
        (?P<int>\d*)          # consisting of a (possibly empty) integer part
        (\.(?P<frac>\d*))?    # followed by an optional fractional part
        (E(?P<exp>[-+]?\d+))? # followed by an optional exponent, or...
    |
        Inf(inity)?           # ...an infinity, or...
    |
        (?P<signal>s)?        # ...an (optionally signaling)
        NaN                   # NaN
        (?P<diag>\d*)         # with (possibly empty) diagnostic information.
    )
#    \s*
    $
""", re.VERBOSE | re.IGNORECASE).match

_all_zeros = re.compile('0*$').match
_exact_half = re.compile('50*$').match
del re


##### Useful Constants (internal use only) ################################

# Reusable defaults
Inf = Decimal('Inf')
negInf = Decimal('-Inf')
NaN = Decimal('NaN')
Dec_0 = Decimal(0)
Dec_p1 = Decimal(1)
Dec_n1 = Decimal(-1)

# Infsign[sign] is infinity w/ that sign
Infsign = (Inf, negInf)



if __name__ == '__main__':
    import doctest, sys
    doctest.testmod(sys.modules[__name__])
