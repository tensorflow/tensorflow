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
"""TensorFlow Debugger (tfdbg) User-Interface Factory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


SUPPORTED_UI_TYPES = ["curses", "readline"]


def get_ui(ui_type, on_ui_exit=None, available_ui_types=None):
  """Create a `base_ui.BaseUI` subtype.

  This factory method attempts to fallback to other available ui_types on
  ImportError. For example, if `ui_type` is `curses`, but `curses` cannot be
  imported properly, e.g., on Windows, will fallback to `readline`.

  Args:
    ui_type: (`str`) requested UI type. Currently supported:
      (curses | readline)
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    available_ui_types: (`None` or `list` of `str`) Manually-set available
      ui_types.

  Returns:
    A `base_ui.BaseUI` subtype object.

  Raises:
    ValueError: on invalid ui_type or on exhausting or fallback ui_types.
  """
  if available_ui_types is None:
    available_ui_types = copy.deepcopy(SUPPORTED_UI_TYPES)

  if ui_type and (ui_type not in available_ui_types):
    raise ValueError("Invalid ui_type: '%s'" % ui_type)

  try:
    # pylint: disable=g-import-not-at-top
    if not ui_type or ui_type == "curses":
      from tensorflow.python.debug.cli import curses_ui
      return curses_ui.CursesUI(on_ui_exit=on_ui_exit)
    elif ui_type == "readline":
      from tensorflow.python.debug.cli import readline_ui
      return readline_ui.ReadlineUI(on_ui_exit=on_ui_exit)
    # pylint: enable=g-import-not-at-top
  except ImportError:
    available_ui_types.remove(ui_type)
    if not available_ui_types:
      raise ValueError("Exhausted all fallback ui_types.")
    return get_ui(available_ui_types[0],
                  available_ui_types=available_ui_types)
