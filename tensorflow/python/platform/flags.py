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

"""Import router for absl.flags. See https://github.com/abseil/abseil-py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as _logging
import sys as _sys

# go/tf-wildcard-import

from absl.flags import *  # pylint: disable=wildcard-import
import six as _six

from tensorflow.python.util import tf_decorator


# Since we wrap absl.flags DEFINE functions, we need to declare this module
# does not affect key flags.
disclaim_key_flags()  # pylint: disable=undefined-variable


_RENAMED_ARGUMENTS = {
    'flag_name': 'name',
    'default_value': 'default',
    'docstring': 'help',
}


def _wrap_define_function(original_function):
  """Wraps absl.flags's define functions so tf.flags accepts old names."""

  def wrapper(*args, **kwargs):
    """Wrapper function that turns old keyword names to new ones."""
    has_old_names = False
    for old_name, new_name in _six.iteritems(_RENAMED_ARGUMENTS):
      if old_name in kwargs:
        has_old_names = True
        value = kwargs.pop(old_name)
        kwargs[new_name] = value
    if has_old_names:
      _logging.warning(
          'Use of the keyword argument names (flag_name, default_value, '
          'docstring) is deprecated, please use (name, default, help) instead.')
    return original_function(*args, **kwargs)

  return tf_decorator.make_decorator(original_function, wrapper)


class _FlagValuesWrapper(object):
  """Wrapper class for absl.flags.FLAGS.

  The difference is that tf.flags.FLAGS implicitly parses flags with sys.argv
  when accessing the FLAGS values before it's explicitly parsed,
  while absl.flags.FLAGS raises an exception.
  """

  def __init__(self, flags_object):
    self.__dict__['__wrapped'] = flags_object

  def __getattribute__(self, name):
    if name == '__dict__':
      return super(_FlagValuesWrapper, self).__getattribute__(name)
    return self.__dict__['__wrapped'].__getattribute__(name)

  def __getattr__(self, name):
    wrapped = self.__dict__['__wrapped']
    # To maintain backwards compatibility, implicitly parse flags when reading
    # a flag.
    if not wrapped.is_parsed():
      wrapped(_sys.argv)
    return wrapped.__getattr__(name)

  def __setattr__(self, name, value):
    return self.__dict__['__wrapped'].__setattr__(name, value)

  def __delattr__(self, name):
    return self.__dict__['__wrapped'].__delattr__(name)

  def __dir__(self):
    return self.__dict__['__wrapped'].__dir__()

  def __getitem__(self, name):
    return self.__dict__['__wrapped'].__getitem__(name)

  def __setitem__(self, name, flag):
    return self.__dict__['__wrapped'].__setitem__(name, flag)

  def __len__(self):
    return self.__dict__['__wrapped'].__len__()

  def __iter__(self):
    return self.__dict__['__wrapped'].__iter__()

  def __str__(self):
    return self.__dict__['__wrapped'].__str__()

  def __call__(self, *args, **kwargs):
    return self.__dict__['__wrapped'].__call__(*args, **kwargs)


# pylint: disable=invalid-name,used-before-assignment
# absl.flags APIs use `default` as the name of the default value argument.
# Allow the following functions continue to accept `default_value`.
DEFINE_string = _wrap_define_function(DEFINE_string)
DEFINE_boolean = _wrap_define_function(DEFINE_boolean)
DEFINE_bool = DEFINE_boolean
DEFINE_float = _wrap_define_function(DEFINE_float)
DEFINE_integer = _wrap_define_function(DEFINE_integer)
# pylint: enable=invalid-name,used-before-assignment

FLAGS = _FlagValuesWrapper(FLAGS)  # pylint: disable=used-before-assignment
