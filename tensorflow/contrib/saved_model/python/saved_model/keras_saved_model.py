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
# pylint: disable=protected-access
"""Utility functions to save/load keras Model to/from SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import saving


# TODO(kathywu): Remove all contrib callers, switch to tf.keras.
save_keras_model = saving.export_saved_model
load_keras_model = saving.load_from_saved_model
