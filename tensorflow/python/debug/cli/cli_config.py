# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Configurations for TensorFlow Debugger (TFDBG) command-line interfaces."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.platform import gfile

RL = debugger_cli_common.RichLine


class CLIConfig(object):
  """Client-facing configurations for TFDBG command-line interfaces."""

  _CONFIG_FILE_NAME = ".tfdbg_config"

  _DEFAULT_CONFIG = [
      ("graph_recursion_depth", 20),
      ("mouse_mode", True),
  ]

  def __init__(self, config_file_path=None):
    self._config_file_path = (config_file_path or
                              self._default_config_file_path())
    self._config = collections.OrderedDict(self._DEFAULT_CONFIG)
    if gfile.Exists(self._config_file_path):
      config = self._load_from_file()
      for key, value in config.items():
        self._config[key] = value
    self._save_to_file()

    self._set_callbacks = dict()

  def get(self, property_name):
    if property_name not in self._config:
      raise KeyError("%s is not a valid property name." % property_name)
    return self._config[property_name]

  def set(self, property_name, property_val):
    """Set the value of a property.

    Supports limitd property value types: `bool`, `int` and `str`.

    Args:
      property_name: Name of the property.
      property_val: Value of the property. If the property has `bool` type and
        this argument has `str` type, the `str` value will be parsed as a `bool`

    Raises:
      ValueError: if a `str` property_value fails to be parsed as a `bool`.
      KeyError: if `property_name` is an invalid property name.
    """
    if property_name not in self._config:
      raise KeyError("%s is not a valid property name." % property_name)

    orig_val = self._config[property_name]
    if isinstance(orig_val, bool):
      if isinstance(property_val, str):
        if property_val.lower() in ("1", "true", "t", "yes", "y", "on"):
          property_val = True
        elif property_val.lower() in ("0", "false", "f", "no", "n", "off"):
          property_val = False
        else:
          raise ValueError(
              "Invalid string value for bool type: %s" % property_val)
      else:
        property_val = bool(property_val)
    elif isinstance(orig_val, int):
      property_val = int(property_val)
    elif isinstance(orig_val, str):
      property_val = str(property_val)
    else:
      raise TypeError("Unsupported property type: %s" % type(orig_val))
    self._config[property_name] = property_val
    self._save_to_file()

    # Invoke set-callback.
    if property_name in self._set_callbacks:
      self._set_callbacks[property_name](self._config)

  def set_callback(self, property_name, callback):
    """Set a set-callback for given property.

    Args:
      property_name: Name of the property.
      callback: The callback as a `callable` of signature:
          def cbk(config):
        where config is the config after it is set to the new value.
        The callback is invoked each time the set() method is called with the
        matching property_name.

    Raises:
      KeyError: If property_name does not exist.
      TypeError: If `callback` is not callable.
    """
    if property_name not in self._config:
      raise KeyError("%s is not a valid property name." % property_name)
    if not callable(callback):
      raise TypeError("The callback object provided is not callable.")
    self._set_callbacks[property_name] = callback

  def _default_config_file_path(self):
    return os.path.join(os.path.expanduser("~"), self._CONFIG_FILE_NAME)

  def _save_to_file(self):
    try:
      with gfile.Open(self._config_file_path, "w") as config_file:
        json.dump(self._config, config_file)
    except IOError:
      pass

  def summarize(self, highlight=None):
    """Get a text summary of the config.

    Args:
      highlight: A property name to highlight in the output.

    Returns:
      A `RichTextLines` output.
    """
    lines = [RL("Command-line configuration:", "bold"), RL("")]
    for name, val in self._config.items():
      highlight_attr = "bold" if name == highlight else None
      line = RL("  ")
      line += RL(name, ["underline", highlight_attr])
      line += RL(": ")
      line += RL(str(val), font_attr=highlight_attr)
      lines.append(line)
    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

  def _load_from_file(self):
    try:
      with gfile.Open(self._config_file_path, "r") as config_file:
        config_dict = json.load(config_file)
        config = collections.OrderedDict()
        for key in sorted(config_dict.keys()):
          config[key] = config_dict[key]
        return config
    except (IOError, ValueError):
      # The reading of the config file may fail due to IO issues or file
      # corruption. We do not want tfdbg to error out just because of that.
      return dict()
