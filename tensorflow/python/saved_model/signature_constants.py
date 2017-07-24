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
"""Signature constants for SavedModel save and restore operations.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.all_util import remove_undocumented


# Key in the signature def map for `default` serving signatures. The default
# signature is used in inference requests where a specific signature was not
# specified.
DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"

################################################################################
# Classification API constants.

# Classification inputs.
CLASSIFY_INPUTS = "inputs"

# Classification method name used in a SignatureDef.
CLASSIFY_METHOD_NAME = "tensorflow/serving/classify"

# Classification classes output.
CLASSIFY_OUTPUT_CLASSES = "classes"

# Classification scores output.
CLASSIFY_OUTPUT_SCORES = "scores"

################################################################################
# Prediction API constants.

# Predict inputs.
PREDICT_INPUTS = "inputs"

# Prediction method name used in a SignatureDef.
PREDICT_METHOD_NAME = "tensorflow/serving/predict"

# Predict outputs.
PREDICT_OUTPUTS = "outputs"

################################################################################
# Regression API constants.

# Regression inputs.
REGRESS_INPUTS = "inputs"

# Regression method name used in a SignatureDef.
REGRESS_METHOD_NAME = "tensorflow/serving/regress"

# Regression outputs.
REGRESS_OUTPUTS = "outputs"

################################################################################


_allowed_symbols = [
    "DEFAULT_SERVING_SIGNATURE_DEF_KEY",
    "CLASSIFY_INPUTS",
    "CLASSIFY_METHOD_NAME",
    "CLASSIFY_OUTPUT_CLASSES",
    "CLASSIFY_OUTPUT_SCORES",
    "PREDICT_INPUTS",
    "PREDICT_METHOD_NAME",
    "PREDICT_OUTPUTS",
    "REGRESS_INPUTS",
    "REGRESS_METHOD_NAME",
    "REGRESS_OUTPUTS",
]
remove_undocumented(__name__, _allowed_symbols)
