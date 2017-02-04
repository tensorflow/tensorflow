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
"""Constants for SavedModel save and restore operations.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.all_util import remove_undocumented

# Subdirectory name containing the asset files.
ASSETS_DIRECTORY = "assets"

# CollectionDef key containing SavedModel assets.
ASSETS_KEY = "saved_model_assets"

# CollectionDef key for the legacy init op.
LEGACY_INIT_OP_KEY = "legacy_init_op"

# CollectionDef key for the SavedModel main op.
MAIN_OP_KEY = "saved_model_main_op"

# Schema version for SavedModel.
SAVED_MODEL_SCHEMA_VERSION = 1

# File name for SavedModel protocol buffer.
SAVED_MODEL_FILENAME_PB = "saved_model.pb"

# File name for text version of SavedModel protocol buffer.
SAVED_MODEL_FILENAME_PBTXT = "saved_model.pbtxt"

# Subdirectory name containing the variables/checkpoint files.
VARIABLES_DIRECTORY = "variables"

# File name used for variables.
VARIABLES_FILENAME = "variables"


_allowed_symbols = [
    "ASSETS_DIRECTORY",
    "ASSETS_KEY",
    "LEGACY_INIT_OP_KEY",
    "MAIN_OP_KEY",
    "SAVED_MODEL_SCHEMA_VERSION",
    "SAVED_MODEL_FILENAME_PB",
    "SAVED_MODEL_FILENAME_PBTXT",
    "VARIABLES_DIRECTORY",
    "VARIABLES_FILENAME",
]
remove_undocumented(__name__, _allowed_symbols)
