"""Implementation of the flags interface."""
import tensorflow.python.platform

import argparse

_global_parser = argparse.ArgumentParser()

class _FlagValues(object):

  def __init__(self):
    """Global container and accessor for flags and their values."""
    self.__dict__['__flags'] = {}
    self.__dict__['__parsed'] = False

  def _parse_flags(self):
    result = _global_parser.parse_args()
    for flag_name, val in vars(result).items():
      self.__dict__['__flags'][flag_name] = val
    self.__dict__['__parsed'] = True

  def __getattr__(self, name):
    """Retrieves the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    if name not in self.__dict__['__flags']:
      raise AttributeError(name)
    return self.__dict__['__flags'][name]

  def __setattr__(self, name, value):
    """Sets the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    self.__dict__['__flags'][name] = value


def _define_helper(flag_name, default_value, docstring, flagtype):
  """Registers 'flag_name' with 'default_value' and 'docstring'."""
  _global_parser.add_argument("--" + flag_name,
                              default=default_value,
                              help=docstring,
                              type=flagtype)


# Provides the global object that can be used to access flags.
FLAGS = _FlagValues()


def DEFINE_string(flag_name, default_value, docstring):
  """Defines a flag of type 'string'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a string.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, str)


def DEFINE_integer(flag_name, default_value, docstring):
  """Defines a flag of type 'int'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as an int.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, int)


def DEFINE_boolean(flag_name, default_value, docstring):
  """Defines a flag of type 'boolean'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a boolean.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, bool)
  _global_parser.add_argument('--no' + flag_name,
                              action='store_false',
                              dest=flag_name)


def DEFINE_float(flag_name, default_value, docstring):
  """Defines a flag of type 'float'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a float.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, float)
