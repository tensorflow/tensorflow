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

from tensorflow.python.saved_model.experimental import pywrap_libexport
from tensorflow.python.util.tf_export import tf_export

# Subdirectory name containing the asset files.
ASSETS_DIRECTORY = pywrap_libexport.ASSETS_DIRECTORY
tf_export(
    "saved_model.ASSETS_DIRECTORY",
    v1=[
        "saved_model.ASSETS_DIRECTORY", "saved_model.constants.ASSETS_DIRECTORY"
    ]).export_constant(__name__, "ASSETS_DIRECTORY")

# Subdirectory name containing unmanaged files from higher-level APIs.
EXTRA_ASSETS_DIRECTORY = pywrap_libexport.EXTRA_ASSETS_DIRECTORY

# CollectionDef key containing SavedModel assets.
ASSETS_KEY = pywrap_libexport.ASSETS_KEY
tf_export(
    "saved_model.ASSETS_KEY",
    v1=["saved_model.ASSETS_KEY",
        "saved_model.constants.ASSETS_KEY"]).export_constant(
            __name__, "ASSETS_KEY")

# CollectionDef key for the legacy init op.
LEGACY_INIT_OP_KEY = pywrap_libexport.LEGACY_INIT_OP_KEY
tf_export(
    v1=[
        "saved_model.LEGACY_INIT_OP_KEY",
        "saved_model.constants.LEGACY_INIT_OP_KEY"
    ]).export_constant(__name__, "LEGACY_INIT_OP_KEY")

# CollectionDef key for the SavedModel main op.
MAIN_OP_KEY = pywrap_libexport.MAIN_OP_KEY
tf_export(
    v1=["saved_model.MAIN_OP_KEY",
        "saved_model.constants.MAIN_OP_KEY"]).export_constant(
            __name__, "MAIN_OP_KEY")

# CollectionDef key for the SavedModel train op.
# Not exported while export_all_saved_models is experimental.
TRAIN_OP_KEY = pywrap_libexport.TRAIN_OP_KEY

# Schema version for SavedModel.
SAVED_MODEL_SCHEMA_VERSION = pywrap_libexport.SAVED_MODEL_SCHEMA_VERSION
tf_export(
    "saved_model.SAVED_MODEL_SCHEMA_VERSION",
    v1=[
        "saved_model.SAVED_MODEL_SCHEMA_VERSION",
        "saved_model.constants.SAVED_MODEL_SCHEMA_VERSION"
    ]).export_constant(__name__, "SAVED_MODEL_SCHEMA_VERSION")

# File name for SavedModel protocol buffer.
SAVED_MODEL_FILENAME_PB = pywrap_libexport.SAVED_MODEL_FILENAME_PB
tf_export(
    "saved_model.SAVED_MODEL_FILENAME_PB",
    v1=[
        "saved_model.SAVED_MODEL_FILENAME_PB",
        "saved_model.constants.SAVED_MODEL_FILENAME_PB"
    ]).export_constant(__name__, "SAVED_MODEL_FILENAME_PB")

# File name for text version of SavedModel protocol buffer.
SAVED_MODEL_FILENAME_PBTXT = pywrap_libexport.SAVED_MODEL_FILENAME_PBTXT
tf_export(
    "saved_model.SAVED_MODEL_FILENAME_PBTXT",
    v1=[
        "saved_model.SAVED_MODEL_FILENAME_PBTXT",
        "saved_model.constants.SAVED_MODEL_FILENAME_PBTXT"
    ]).export_constant(__name__, "SAVED_MODEL_FILENAME_PBTXT")

# Subdirectory where debugging related files are written.
DEBUG_DIRECTORY = pywrap_libexport.DEBUG_DIRECTORY
tf_export(
    "saved_model.DEBUG_DIRECTORY",
    v1=[
        "saved_model.DEBUG_DIRECTORY",
        "saved_model.constants.DEBUG_DIRECTORY",
    ]).export_constant(__name__, "DEBUG_DIRECTORY")

# File name for GraphDebugInfo protocol buffer which corresponds to the
# SavedModel.
DEBUG_INFO_FILENAME_PB = pywrap_libexport.DEBUG_INFO_FILENAME_PB
tf_export(
    "saved_model.DEBUG_INFO_FILENAME_PB",
    v1=[
        "saved_model.DEBUG_INFO_FILENAME_PB",
        "saved_model.constants.DEBUG_INFO_FILENAME_PB"
    ]).export_constant(__name__, "DEBUG_INFO_FILENAME_PB")

# Subdirectory name containing the variables/checkpoint files.
VARIABLES_DIRECTORY = pywrap_libexport.VARIABLES_DIRECTORY
tf_export(
    "saved_model.VARIABLES_DIRECTORY",
    v1=[
        "saved_model.VARIABLES_DIRECTORY",
        "saved_model.constants.VARIABLES_DIRECTORY"
    ]).export_constant(__name__, "VARIABLES_DIRECTORY")

# File name used for variables.
VARIABLES_FILENAME = pywrap_libexport.VARIABLES_FILENAME
tf_export(
    "saved_model.VARIABLES_FILENAME",
    v1=[
        "saved_model.VARIABLES_FILENAME",
        "saved_model.constants.VARIABLES_FILENAME"
    ]).export_constant(__name__, "VARIABLES_FILENAME")

# The initialization and train ops for a MetaGraph are stored in the
# signature def map. The ops are added to the map with the following keys.
INIT_OP_SIGNATURE_KEY = pywrap_libexport.INIT_OP_SIGNATURE_KEY
TRAIN_OP_SIGNATURE_KEY = pywrap_libexport.TRAIN_OP_SIGNATURE_KEY
