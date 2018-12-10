# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for tf.data options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _internal_attr_name(name):
  return "_" + name


class OptionsBase(object):
  """Base class for representing a set of tf.data options.

  Attributes:
    _options: Stores the option values.
  """

  def __init__(self):
    self._options = {}

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return NotImplemented
    for name in set(self._options) | set(other._options):  # pylint: disable=protected-access
      if getattr(self, name) != getattr(other, name):
        return False
    return True

  def __ne__(self, other):
    if isinstance(other, self.__class__):
      return not self.__eq__(other)
    else:
      return NotImplemented


def create_option(name, ty, docstring, default=None):
  """Creates a type-checked property.

  Args:
    name: the name to use
    ty: the type to use
    docstring: the docstring to use
    default: the default value to use

  Returns:
    A type-checked property.
  """

  def get_fn(self):
    return self._options.get(name, default)  # pylint: disable=protected-access

  def set_fn(self, value):
    if not isinstance(value, ty):
      raise TypeError("Property \"%s\" must be of type %s, got: %r (type: %r)" %
                      (name, ty, value, type(value)))
    self._options[name] = value  # pylint: disable=protected-access

  return property(get_fn, set_fn, None, docstring)


def merge_options(*options_list):
  """Merges the given options, returning the result as a new options object.

  The input arguments are expected to have a matching type that derives from
  `OptionsBase` (and thus each represent a set of options). The method outputs
  an object of the same type created by merging the sets of options represented
  by the input arguments.

  The sets of options can be merged as long as there does not exist an option
  with different non-default values.

  If an option is an instance of `OptionsBase` itself, then this method is
  applied recursively to the set of options represented by this option.

  Args:
    *options_list: options to merge

  Raises:
    TypeError: if the input arguments are incompatible or not derived from
      `OptionsBase`
    ValueError: if the given options cannot be merged

  Returns:
    A new options object which is the result of merging the given options.
  """
  if len(options_list) < 1:
    raise ValueError("At least one options should be provided")
  result_type = type(options_list[0])

  for options in options_list:
    if not isinstance(options, result_type):
      raise TypeError("Incompatible options type: %r vs %r" % (type(options),
                                                               result_type))

  if not isinstance(options_list[0], OptionsBase):
    raise TypeError("The inputs should inherit from `OptionsBase`")

  default_options = result_type()
  result = result_type()
  for options in options_list:
    # Iterate over all set options and merge the into the result.
    for name in options._options:  # pylint: disable=protected-access
      this = getattr(result, name)
      that = getattr(options, name)
      default = getattr(default_options, name)
      if that == default:
        continue
      elif this == default:
        setattr(result, name, that)
      elif isinstance(this, OptionsBase):
        setattr(result, name, merge_options(this, that))
      elif this != that:
        raise ValueError(
            "Cannot merge incompatible values (%r and %r) of option: %s" %
            (this, that, name))
  return result
