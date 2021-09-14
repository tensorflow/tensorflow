# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tensor utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect
import re

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls


# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


class DeprecatedNamesAlreadySet(Exception):
  """Raised when setting deprecated names multiple times for the same symbol."""
  pass


def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated functions."""
  main_text = ['THIS FUNCTION IS DEPRECATED. It will be removed %s.' %
               ('in a future version' if date is None else ('after %s' % date))]
  if instructions:
    main_text.append('Instructions for updating:')
  return decorator_utils.add_notice_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION',
      '(deprecated)', main_text)


def _add_deprecated_arg_notice_to_docstring(doc, date, instructions,
                                            deprecated_names):
  """Adds a deprecation notice to a docstring for deprecated arguments."""

  deprecation_string = ', '.join(sorted(deprecated_names))

  return decorator_utils.add_notice_to_docstring(
      doc, instructions, 'DEPRECATED FUNCTION ARGUMENTS',
      '(deprecated arguments)', [
          'SOME ARGUMENTS ARE DEPRECATED: `(%s)`. '
          'They will be removed %s.' %
          (deprecation_string, 'in a future version' if date is None else
           ('after %s' % date)), 'Instructions for updating:'
      ])


def _add_deprecated_arg_value_notice_to_docstring(doc, date, instructions,
                                                  deprecated_name_value_dict):
  """Adds a deprecation notice to a docstring for deprecated arguments."""

  deprecation_string = ', '.join(
      '%s=%r' % (key, value)
      for key, value in sorted(deprecated_name_value_dict.items()))

  when = 'in a future version' if date is None else ('after %s' % date)

  return decorator_utils.add_notice_to_docstring(
      doc, instructions, 'DEPRECATED FUNCTION ARGUMENT VALUES',
      '(deprecated argument values)', [
          'SOME ARGUMENT VALUES ARE DEPRECATED: `(%s)`. '
          'They will be removed %s.' % (deprecation_string, when),
          'Instructions for updating:'
      ])


def _validate_deprecation_args(date, instructions):
  if date is not None and not re.match(r'20\d\d-[01]\d-[0123]\d', date):
    raise ValueError(f'Date must be in format YYYY-MM-DD. Received: {date}')
  if not instructions:
    raise ValueError(
        'Don\'t deprecate things without conversion instructions! Specify '
        'the `instructions` argument.')


def _call_location(outer=False):
  """Returns call location given level up from current call."""
  # Two up: <_call_location>, <_call_location's caller>
  # tf_inspect is not required here. Please ignore the lint warning by adding
  # DISABLE_IMPORT_INSPECT_CHECK=TRUE to your cl description. Using it caused
  # test timeouts (b/189384061).
  f = inspect.currentframe().f_back.f_back
  parent = f and f.f_back
  if outer and parent is not None:
    f = parent
  return '{}:{}'.format(f.f_code.co_filename, f.f_lineno)


def _safe_eq(a, b):
  if a is None or b is None:
    return a is None and b is None
  return a == b


def _wrap_decorator(wrapped_function):
  """Indicate that one function wraps another.

  This decorator wraps a function using `tf_decorator.make_decorator`
  so that doc generation scripts can pick up original function
  signature.
  It would be better to use @functools.wrap decorator, but it would
  not update function signature to match wrapped function in Python 2.

  Args:
    wrapped_function: The function that decorated function wraps.

  Returns:
    Function that accepts wrapper function as an argument and returns
    `TFDecorator` instance.
  """
  def wrapper(wrapper_func):
    return tf_decorator.make_decorator(wrapped_function, wrapper_func)
  return wrapper


def deprecated_alias(deprecated_name, name, func_or_class, warn_once=True):
  """Deprecate a symbol in favor of a new name with identical semantics.

  This function is meant to be used when defining a backwards-compatibility
  alias for a symbol which has been moved. For example:

  module1.py:
  ```python
  class NewNameForClass: pass
  ```

  module2.py:
  ```python
  import module1

  DeprecatedNameForClass = deprecated_alias(
    deprecated_name='module2.DeprecatedNameForClass',
    name='module1.NewNameForClass',
    func_or_class=module1.NewNameForClass)
  ```

  This function works for classes and functions.

  For classes, it creates a new class which is functionally identical (it
  inherits from the original, and overrides its constructor), but which prints
  a deprecation warning when an instance is created. It also adds a deprecation
  notice to the class' docstring.

  For functions, it returns a function wrapped by `tf_decorator.make_decorator`.
  That function prints a warning when used, and has a deprecation notice in its
  docstring. This is more or less equivalent (the deprecation warning has
  slightly different text) to writing:

  ```python
  @deprecated
  def deprecated_alias(original_args):
    real_function(original_args)
  ```

  Args:
    deprecated_name: The name of the symbol that is being deprecated, to be used
      in the warning message. This should be its fully qualified name to avoid
      confusion.
    name: The name of the symbol that is to be used instead of the deprecated
      name. This should be a fully qualified name to avoid confusion.
    func_or_class: The (non-deprecated) class or function for which a deprecated
      alias should be created.
    warn_once: If True (the default), only print a deprecation warning the first
      time this function is used, or the class is instantiated.

  Returns:
    A wrapped version of `func_or_class` which prints a deprecation warning on
    use and has a modified docstring.
  """
  if tf_inspect.isclass(func_or_class):

    # Make a new class with __init__ wrapped in a warning.
    class _NewClass(func_or_class):  # pylint: disable=missing-docstring
      __doc__ = decorator_utils.add_notice_to_docstring(
          func_or_class.__doc__, 'Please use %s instead.' % name,
          'DEPRECATED CLASS',
          '(deprecated)', ['THIS CLASS IS DEPRECATED. '
                           'It will be removed in a future version. '])
      __name__ = func_or_class.__name__
      __module__ = _call_location(outer=True)

      @_wrap_decorator(func_or_class.__init__)
      def __init__(self, *args, **kwargs):
        if hasattr(_NewClass.__init__, '__func__'):
          # Python 2
          _NewClass.__init__.__func__.__doc__ = func_or_class.__init__.__doc__
        else:
          # Python 3
          _NewClass.__init__.__doc__ = func_or_class.__init__.__doc__

        if _PRINT_DEPRECATION_WARNINGS:
          # We're making the alias as we speak. The original may have other
          # aliases, so we cannot use it to check for whether it's already been
          # warned about.
          if _NewClass.__init__ not in _PRINTED_WARNING:
            if warn_once:
              _PRINTED_WARNING[_NewClass.__init__] = True
            logging.warning(
                'From %s: The name %s is deprecated. Please use %s instead.\n',
                _call_location(), deprecated_name, name)
        super(_NewClass, self).__init__(*args, **kwargs)

    return _NewClass
  else:
    decorator_utils.validate_callable(func_or_class, 'deprecated')

    # Make a wrapper for the original
    @functools.wraps(func_or_class)
    def new_func(*args, **kwargs):  # pylint: disable=missing-docstring
      if _PRINT_DEPRECATION_WARNINGS:
        # We're making the alias as we speak. The original may have other
        # aliases, so we cannot use it to check for whether it's already been
        # warned about.
        if new_func not in _PRINTED_WARNING:
          if warn_once:
            _PRINTED_WARNING[new_func] = True
          logging.warning(
              'From %s: The name %s is deprecated. Please use %s instead.\n',
              _call_location(), deprecated_name, name)
      return func_or_class(*args, **kwargs)
    return tf_decorator.make_decorator(
        func_or_class, new_func, 'deprecated',
        _add_deprecated_function_notice_to_docstring(
            func_or_class.__doc__, None, 'Please use %s instead.' % name))


def deprecated_endpoints(*args):
  """Decorator for marking endpoints deprecated.

  This decorator does not print deprecation messages.
  TODO(annarev): eventually start printing deprecation warnings when
  @deprecation_endpoints decorator is added.

  Args:
    *args: Deprecated endpoint names.

  Returns:
    A function that takes symbol as an argument and adds
    _tf_deprecated_api_names to that symbol.
    _tf_deprecated_api_names would be set to a list of deprecated
    endpoint names for the symbol.
  """
  def deprecated_wrapper(func):
    # pylint: disable=protected-access
    if '_tf_deprecated_api_names' in func.__dict__:
      raise DeprecatedNamesAlreadySet(
          f'Cannot set deprecated names for {func.__name__} to {args}. '
          'Deprecated names are already set to '
          f'{func._tf_deprecated_api_names}.')
    func._tf_deprecated_api_names = args
    # pylint: disable=protected-access
    return func
  return deprecated_wrapper


def deprecated(date, instructions, warn_once=True):
  """Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  """
  _validate_deprecation_args(date, instructions)

  def deprecated_wrapper(func_or_class):
    """Deprecation wrapper."""
    if isinstance(func_or_class, type):
      # If a class is deprecated, you actually want to wrap the constructor.
      cls = func_or_class
      if cls.__new__ is object.__new__:
        func = cls.__init__
        constructor_name = '__init__'
      else:
        func = cls.__new__
        constructor_name = '__new__'

    else:
      cls = None
      constructor_name = None
      func = func_or_class

    decorator_utils.validate_callable(func, 'deprecated')
    @functools.wraps(func)
    def new_func(*args, **kwargs):  # pylint: disable=missing-docstring
      if _PRINT_DEPRECATION_WARNINGS:
        if func not in _PRINTED_WARNING:
          if warn_once:
            _PRINTED_WARNING[func] = True
          logging.warning(
              'From %s: %s (from %s) is deprecated and will be removed %s.\n'
              'Instructions for updating:\n%s',
              _call_location(), decorator_utils.get_qualified_name(func),
              func.__module__,
              'in a future version' if date is None else ('after %s' % date),
              instructions)
      return func(*args, **kwargs)

    doc_controls.set_deprecated(new_func)
    new_func = tf_decorator.make_decorator(
        func, new_func, 'deprecated',
        _add_deprecated_function_notice_to_docstring(func.__doc__, date,
                                                     instructions))

    if cls is None:
      return new_func
    else:
      # Insert the wrapped function as the constructor
      setattr(cls, constructor_name, new_func)

      # And update the docstring of the class.
      cls.__doc__ = _add_deprecated_function_notice_to_docstring(
          cls.__doc__, date, instructions)

      return cls

  return deprecated_wrapper


DeprecatedArgSpec = collections.namedtuple(
    'DeprecatedArgSpec', ['position', 'has_ok_value', 'ok_value'])


def deprecated_args(date, instructions, *deprecated_arg_names_or_tuples,
                    **kwargs):
  """Decorator for marking specific function arguments as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument. It has the following format:

    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> includes the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    *deprecated_arg_names_or_tuples: String or 2-Tuple (String,
      ok_val).  The string is the deprecated argument name.
      Optionally, an ok-value may be provided.  If the user provided
      argument equals this value, the warning is suppressed.
    **kwargs: If `warn_once=False` is passed, every call with a deprecated
      argument will log a warning. The default behavior is to only warn the
      first time the function is called with any given deprecated argument.
      All other kwargs raise `ValueError`.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, instructions are
      empty, the deprecated arguments are not present in the function
      signature, the second element of a deprecated_tuple is not a
      list, or if a kwarg other than `warn_once` is passed.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_arg_names_or_tuples:
    raise ValueError('Specify which argument is deprecated.')
  if kwargs and list(kwargs.keys()) != ['warn_once']:
    kwargs.pop('warn_once', None)
    raise ValueError(f'Illegal argument passed to deprecated_args: {kwargs}')
  warn_once = kwargs.get('warn_once', True)

  def _get_arg_names_to_ok_vals():
    """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
    d = {}
    for name_or_tuple in deprecated_arg_names_or_tuples:
      if isinstance(name_or_tuple, tuple):
        d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])
      else:
        d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)
    return d

  def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
    """Builds a dictionary from deprecated arguments to their spec.

    Returned dict is keyed by argument name.
    Each value is a DeprecatedArgSpec with the following fields:
       position: The zero-based argument position of the argument
         within the signature.  None if the argument isn't found in
         the signature.
       ok_values:  Values of this argument for which warning will be
         suppressed.

    Args:
      names_to_ok_vals: dict from string arg_name to a list of values,
        possibly empty, which should not elicit a warning.
      arg_spec: Output from tf_inspect.getfullargspec on the called function.

    Returns:
      Dictionary from arg_name to DeprecatedArgSpec.
    """
    # Extract argument list
    arg_space = arg_spec.args + arg_spec.kwonlyargs
    arg_name_to_pos = {
        name: pos for pos, name in enumerate(arg_space)}
    deprecated_positional_args = {}
    for arg_name, spec in iter(names_to_ok_vals.items()):
      if arg_name in arg_name_to_pos:
        pos = arg_name_to_pos[arg_name]
        deprecated_positional_args[arg_name] = DeprecatedArgSpec(
            pos, spec.has_ok_value, spec.ok_value)
    return deprecated_positional_args

  deprecated_arg_names = _get_arg_names_to_ok_vals()

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_args')

    arg_spec = tf_inspect.getfullargspec(func)
    deprecated_positions = _get_deprecated_positional_arguments(
        deprecated_arg_names, arg_spec)

    is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
    is_kwargs_deprecated = arg_spec.varkw in deprecated_arg_names

    if (len(deprecated_positions) + is_varargs_deprecated
        + is_kwargs_deprecated
        != len(deprecated_arg_names_or_tuples)):
      known_args = (arg_spec.args
                    + arg_spec.kwonlyargs
                    + [arg_spec.varargs, arg_spec.varkw])
      missing_args = [arg_name for arg_name in deprecated_arg_names
                      if arg_name not in known_args]
      raise ValueError('The following deprecated arguments are not present '
                       f'in the function signature: {missing_args}. '
                       'Expected arguments from the following list: '
                       f'{known_args}.')

    def _same_value(a, b):
      """A comparison operation that works for multiple object types.

      Returns True for two empty lists, two numeric values with the
      same value, etc.

      Returns False for (pd.DataFrame, None), and other pairs which
      should not be considered equivalent.

      Args:
        a: value one of the comparison.
        b: value two of the comparison.

      Returns:
        A boolean indicating whether the two inputs are the same value
        for the purposes of deprecation.
      """
      if a is b:
        return True
      try:
        equality = a == b
        if isinstance(equality, bool):
          return equality
      except TypeError:
        return False
      return False

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      # TODO(apassos) figure out a way to have reasonable performance with
      # deprecation warnings and eager mode.
      if is_in_graph_mode.IS_IN_GRAPH_MODE() and _PRINT_DEPRECATION_WARNINGS:
        invalid_args = []
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, spec in iter(deprecated_positions.items()):
          if (spec.position < len(args) and
              not (spec.has_ok_value and
                   _same_value(named_args[arg_name], spec.ok_value))):
            invalid_args.append(arg_name)
        if is_varargs_deprecated and len(args) > len(arg_spec.args):
          invalid_args.append(arg_spec.varargs)
        if is_kwargs_deprecated and kwargs:
          invalid_args.append(arg_spec.varkw)
        for arg_name in deprecated_arg_names:
          if (arg_name in kwargs and
              not (deprecated_positions[arg_name].has_ok_value and
                   _same_value(named_args[arg_name],
                               deprecated_positions[arg_name].ok_value))):
            invalid_args.append(arg_name)
        for arg_name in invalid_args:
          if (func, arg_name) not in _PRINTED_WARNING:
            if warn_once:
              _PRINTED_WARNING[(func, arg_name)] = True
            logging.warning(
                'From %s: calling %s (from %s) with %s is deprecated and will '
                'be removed %s.\nInstructions for updating:\n%s',
                _call_location(), decorator_utils.get_qualified_name(func),
                func.__module__, arg_name,
                'in a future version' if date is None else ('after %s' % date),
                instructions)
      return func(*args, **kwargs)

    doc = _add_deprecated_arg_notice_to_docstring(
        func.__doc__, date, instructions, sorted(deprecated_arg_names.keys()))
    return tf_decorator.make_decorator(func, new_func, 'deprecated', doc)

  return deprecated_wrapper


def deprecated_arg_values(date, instructions, warn_once=True,
                          **deprecated_kwargs):
  """Decorator for marking specific function argument values as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument values. It has the following format:

    Calling <function> (from <module>) with <arg>=<value> is deprecated and
    will be removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: If `True`, warn only the first time this function is called with
      deprecated argument values. Otherwise, every call (with a deprecated
      argument value) will log a warning.
    **deprecated_kwargs: The deprecated argument values.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_kwargs:
    raise ValueError('Specify which argument values are deprecated.')

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_arg_values')
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      if _PRINT_DEPRECATION_WARNINGS:
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, arg_value in deprecated_kwargs.items():
          if arg_name in named_args and _safe_eq(named_args[arg_name],
                                                 arg_value):
            if (func, arg_name) not in _PRINTED_WARNING:
              if warn_once:
                _PRINTED_WARNING[(func, arg_name)] = True
              logging.warning(
                  'From %s: calling %s (from %s) with %s=%s is deprecated and '
                  'will be removed %s.\nInstructions for updating:\n%s',
                  _call_location(), decorator_utils.get_qualified_name(func),
                  func.__module__, arg_name, arg_value, 'in a future version'
                  if date is None else ('after %s' % date), instructions)
      return func(*args, **kwargs)

    doc = _add_deprecated_arg_value_notice_to_docstring(
        func.__doc__, date, instructions, deprecated_kwargs)
    return tf_decorator.make_decorator(func, new_func, 'deprecated', doc)

  return deprecated_wrapper


def deprecated_argument_lookup(new_name, new_value, old_name, old_value):
  """Looks up deprecated argument name and ensures both are not used.

  Args:
    new_name: new name of argument
    new_value: value of new argument (or None if not used)
    old_name: old name of argument
    old_value: value of old argument (or None if not used)
  Returns:
    The effective argument that should be used.
  Raises:
    ValueError: if new_value and old_value are both non-null
  """
  if old_value is not None:
    if new_value is not None:
      raise ValueError(f"Cannot specify both '{old_name}' and '{new_name}'.")
    return old_value
  return new_value


def rewrite_argument_docstring(old_doc, old_argument, new_argument):
  return old_doc.replace('`%s`' % old_argument, '`%s`' % new_argument).replace(
      '%s:' % old_argument, '%s:' % new_argument)


@tf_contextlib.contextmanager
def silence():
  """Temporarily silence deprecation warnings."""
  global _PRINT_DEPRECATION_WARNINGS
  print_deprecation_warnings = _PRINT_DEPRECATION_WARNINGS
  _PRINT_DEPRECATION_WARNINGS = False
  yield
  _PRINT_DEPRECATION_WARNINGS = print_deprecation_warnings


class HiddenTfApiAttribute(property):
  """Hides a class attribute from the public API.

  Attributes in public classes can be hidden from the API by having an '_' in
  front of the name (e.g. ClassName._variables). This doesn't work when
  attributes or methods are inherited from a parent class. To hide inherited
  attributes, set their values to be `deprecation.hide_attribute_from_api`.
  For example, this is used in V2 Estimator to hide the deprecated
  export_savedmodel method:
    class EstimatorV2(Estimator):
       export_savedmodel = deprecation.hide_attribute_from_api('...')
  """

  def __init__(self, deprecation_message):

    def raise_error(unused_self):
      raise AttributeError(deprecation_message)

    super(HiddenTfApiAttribute, self).__init__(raise_error)


hide_attribute_from_api = HiddenTfApiAttribute  # pylint: disable=invalid-name

# TODO(kathywu): Remove once cl/246395236 is submitted.
HIDDEN_ATTRIBUTE = HiddenTfApiAttribute('This attribute has been deprecated.')
