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

from tensorflow.python.util.tf_export import tf_export


# Key in the signature def map for `default` serving signatures. The default
# signature is used in inference requests where a specific signature was not
# specified.
DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"
tf_export("saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY"
         ).export_constant(__name__, "DEFAULT_SERVING_SIGNATURE_DEF_KEY")

################################################################################
# Classification API constants.

# Classification inputs.
CLASSIFY_INPUTS = "inputs"
tf_export("saved_model.signature_constants.CLASSIFY_INPUTS").export_constant(
    __name__, "CLASSIFY_INPUTS")

# Classification method name used in a SignatureDef.
CLASSIFY_METHOD_NAME = "tensorflow/serving/classify"
tf_export(
    "saved_model.signature_constants.CLASSIFY_METHOD_NAME").export_constant(
        __name__, "CLASSIFY_METHOD_NAME")

# Classification classes output.
CLASSIFY_OUTPUT_CLASSES = "classes"
tf_export(
    "saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES").export_constant(
        __name__, "CLASSIFY_OUTPUT_CLASSES")

# Classification scores output.
CLASSIFY_OUTPUT_SCORES = "scores"
tf_export(
    "saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES").export_constant(
        __name__, "CLASSIFY_OUTPUT_SCORES")

################################################################################
# Prediction API constants.

# Predict inputs.
PREDICT_INPUTS = "inputs"
tf_export("saved_model.signature_constants.PREDICT_INPUTS").export_constant(
    __name__, "PREDICT_INPUTS")

# Prediction method name used in a SignatureDef.
PREDICT_METHOD_NAME = "tensorflow/serving/predict"
tf_export(
    "saved_model.signature_constants.PREDICT_METHOD_NAME").export_constant(
        __name__, "PREDICT_METHOD_NAME")

# Predict outputs.
PREDICT_OUTPUTS = "outputs"
tf_export("saved_model.signature_constants.PREDICT_OUTPUTS").export_constant(
    __name__, "PREDICT_OUTPUTS")

################################################################################
# Regression API constants.

# Regression inputs.
REGRESS_INPUTS = "inputs"
tf_export("saved_model.signature_constants.REGRESS_INPUTS").export_constant(
    __name__, "REGRESS_INPUTS")

# Regression method name used in a SignatureDef.
REGRESS_METHOD_NAME = "tensorflow/serving/regress"
tf_export(
    "saved_model.signature_constants.REGRESS_METHOD_NAME").export_constant(
        __name__, "REGRESS_METHOD_NAME")

# Regression outputs.
REGRESS_OUTPUTS = "outputs"
tf_export("saved_model.signature_constants.REGRESS_OUTPUTS").export_constant(
    __name__, "REGRESS_OUTPUTS")

################################################################################
