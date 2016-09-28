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

ASSETS_DIRECTORY = "assets"
ASSETS_KEY = "saved_model_assets"

SAVED_MODEL_SCHEMA_VERSION = 1
SAVED_MODEL_FILENAME_PB = "saved_model.pb"
SAVED_MODEL_FILENAME_PBTXT = "saved_model.pbtxt"

TAG_SERVING = "serve"
TAG_TRAINING = "train"

VARIABLES_DIRECTORY = "variables"
VARIABLES_FILENAME = "saved_model_variables"
VARIABLES_FILENAME_SHARDED = VARIABLES_FILENAME + "-?????-of-?????"
