# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Upgrader for Python scripts from 1.* to 2.0 TensorFlow using SAFETY mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    self.function_keyword_renames = {}
    self.symbol_renames = {}
    self.change_to_function = {}
    self.function_reorders = {}
    self.function_warnings = {}
    self.function_transformers = {}
    self.module_deprecations = module_deprecations_v2.MODULE_DEPRECATIONS

    ## Inform about the addons mappings
    for symbol, replacement in all_renames_v2.addons_symbol_mappings.items():
      warning = (
          ast_edits.WARNING, (
              "(Manual edit required) `{}` has been migrated to `{}` in "
              "TensorFlow Addons. The API spec may have changed during the "
              "migration. Please see https://github.com/tensorflow/addons "
              "for more info.").format(symbol, replacement))
      self.function_warnings[symbol] = warning

    # List module renames. If changed, please update max_submodule_depth.
    self.import_renames = {
        "tensorflow.google":
            ast_edits.ImportRename(
                "tensorflow.google.compat.v1",
                excluded_prefixes=[
                    "tensorflow.google.compat.v1",
                    "tensorflow.google.compat.v2",
                ],
            ),
        "tensorflow":
            ast_edits.ImportRename(
                "tensorflow.compat.v1",
                excluded_prefixes=[
                    "tensorflow.contrib", "tensorflow.flags",
                    "tensorflow.compat.v1", "tensorflow.compat.v2",
                    "tensorflow.google"
                ],
            )
    }
    # Needs to be updated if self.import_renames is changed.
    self.max_submodule_depth = 2

    # TODO(kaftan,annarev): specify replacement from TensorFlow import to
    # compat.v1 import.
