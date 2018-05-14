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

from tensorflow.python.util.tf_export import tf_export

# Subdirectory name containing the asset files.
ASSETS_DIRECTORY = "assets"
tf_export("saved_model.constants.ASSETS_DIRECTORY").export_constant(
    __name__, "ASSETS_DIRECTORY")

# CollectionDef key containing SavedModel assets.
ASSETS_KEY = "saved_model_assets"
tf_export("saved_model.constants.ASSETS_KEY").export_constant(
    __name__, "ASSETS_KEY")

# CollectionDef key for the legacy init op.
LEGACY_INIT_OP_KEY = "legacy_init_op"
tf_export("saved_model.constants.LEGACY_INIT_OP_KEY").export_constant(
    __name__, "LEGACY_INIT_OP_KEY")

# CollectionDef key for the SavedModel main op.
MAIN_OP_KEY = "saved_model_main_op"
tf_export("saved_model.constants.MAIN_OP_KEY").export_constant(
    __name__, "MAIN_OP_KEY")

# CollectionDef key for the SavedModel train op.
# Not exported while export_all_saved_models is in contrib.
TRAIN_OP_KEY = "saved_model_train_op"

# Schema version for SavedModel.
SAVED_MODEL_SCHEMA_VERSION = 1
tf_export("saved_model.constants.SAVED_MODEL_SCHEMA_VERSION").export_constant(
    __name__, "SAVED_MODEL_SCHEMA_VERSION")

# File name for SavedModel protocol buffer.
SAVED_MODEL_FILENAME_PB = "saved_model.pb"
tf_export("saved_model.constants.SAVED_MODEL_FILENAME_PB").export_constant(
    __name__, "SAVED_MODEL_FILENAME_PB")

# File name for text version of SavedModel protocol buffer.
SAVED_MODEL_FILENAME_PBTXT = "saved_model.pbtxt"
tf_export("saved_model.constants.SAVED_MODEL_FILENAME_PBTXT").export_constant(
    __name__, "SAVED_MODEL_FILENAME_PBTXT")

# Subdirectory name containing the variables/checkpoint files.
VARIABLES_DIRECTORY = "variables"
tf_export("saved_model.constants.VARIABLES_DIRECTORY").export_constant(
    __name__, "VARIABLES_DIRECTORY")

# File name used for variables.
VARIABLES_FILENAME = "variables"
tf_export("saved_model.constants.VARIABLES_FILENAME").export_constant(
    __name__, "VARIABLES_FILENAME")


