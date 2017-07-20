# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Implementation of the flags interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse as _argparse

from tensorflow.python.platform import tf_logging as _logging
from tensorflow.python.util.all_util import remove_undocumented

_global_parser = _argparse.ArgumentParser()


# pylint: disable=invalid-name


class _FlagValues(object):
  """Global container and accessor for flags and their values."""

  def __init__(self):
    self.__dict__['__flags'] = {}
    self.__dict__['__parsed'] = False
    self.__dict__['__required_flags'] = set()

  def _parse_flags(self, args=None):
    result, unparsed = _global_parser.parse_known_args(args=args)
    for flag_name, val in vars(result).items():
      self.__dict__['__flags'][flag_name] = val
    self.__dict__['__parsed'] = True
    self._assert_all_required()
    return unparsed

  def __getattr__(self, name):
    """Retrieves the 'value' attribute of the flag --name."""
    try:
      parsed = self.__dict__['__parsed']
    except KeyError:
      # May happen during pickle.load or copy.copy
      raise AttributeError(name)
    if not parsed:
      self._parse_flags()
    if name not in self.__dict__['__flags']:
      raise AttributeError(name)
    return self.__dict__['__flags'][name]

  def __setattr__(self, name, value):
    """Sets the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    self.__dict__['__flags'][name] = value
    self._assert_required(name)

  def _add_required_flag(self, item):
    self.__dict__['__required_flags'].add(item)

  def _assert_required(self, flag_name):
    if (flag_name not in self.__dict__['__flags'] or
        self.__dict__['__flags'][flag_name] is None):
      raise AttributeError('Flag --%s must be specified.' % flag_name)

  def _assert_all_required(self):
    for flag_name in self.__dict__['__required_flags']:
      self._assert_required(flag_name)


def _define_helper(flag_name, default_value, docstring, flagtype):
  """Registers 'flag_name' with 'default_value' and 'docstring'."""
  _global_parser.add_argument('--' + flag_name,
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
  # Register a custom function for 'bool' so --flag=True works.
  def str2bool(v):
    return v.lower() in ('true', 't', '1')
  _global_parser.add_argument('--' + flag_name,
                              nargs='?',
                              const=True,
                              help=docstring,
                              default=default_value,
                              type=str2bool)

  # Add negated version, stay consistent with argparse with regard to
  # dashes in flag names.
  _global_parser.add_argument('--no' + flag_name,
                              action='store_false',
                              dest=flag_name.replace('-', '_'))


# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = DEFINE_boolean  # pylint: disable=invalid-name


def DEFINE_float(flag_name, default_value, docstring):
  """Defines a flag of type 'float'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a float.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, float)


def mark_flag_as_required(flag_name):
  """Ensures that flag is not None during program execution.
  
  It is recommended to call this method like this:
  
    if __name__ == '__main__':
      tf.flags.mark_flag_as_required('your_flag_name')
      tf.app.run()
  
  Args:
    flag_name: string, name of the flag to mark as required.
 
  Raises:
    AttributeError: if flag_name is not registered as a valid flag name.
      NOTE: The exception raised will change in the future. 
  """
  if _global_parser.get_default(flag_name) is not None:
    _logging.warn(
        'Flag %s has a non-None default value; therefore, '
        'mark_flag_as_required will pass even if flag is not specified in the '
        'command line!' % flag_name)
  FLAGS._add_required_flag(flag_name)


def mark_flags_as_required(flag_names):
  """Ensures that flags are not None during program execution.
  
  Recommended usage:
  
    if __name__ == '__main__':
      tf.flags.mark_flags_as_required(['flag1', 'flag2', 'flag3'])
      tf.app.run()
  
  Args:
    flag_names: a list/tuple of flag names to mark as required.

  Raises:
    AttributeError: If any of flag name has not already been defined as a flag.
      NOTE: The exception raised will change in the future.
  """
  for flag_name in flag_names:
    mark_flag_as_required(flag_name)


_allowed_symbols = [
    # We rely on gflags documentation.
    'DEFINE_bool',
    'DEFINE_boolean',
    'DEFINE_float',
    'DEFINE_integer',
    'DEFINE_string',
    'FLAGS',
    'mark_flag_as_required',
    'mark_flags_as_required',
]
remove_undocumented(__name__, _allowed_symbols)
