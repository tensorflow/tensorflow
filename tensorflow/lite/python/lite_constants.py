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
"""Constants for TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export

# Enum types from the protobuf promoted to the API
FLOAT = _types_pb2.FLOAT
INT32 = _types_pb2.INT32
INT64 = _types_pb2.INT64
STRING = _types_pb2.STRING
QUANTIZED_UINT8 = _types_pb2.QUANTIZED_UINT8
COMPLEX64 = _types_pb2.COMPLEX64
TENSORFLOW_GRAPHDEF = _toco_flags_pb2.TENSORFLOW_GRAPHDEF
TFLITE = _toco_flags_pb2.TFLITE
GRAPHVIZ_DOT = _toco_flags_pb2.GRAPHVIZ_DOT

_tf_export("lite.constants.FLOAT").export_constant(__name__, "FLOAT")
_tf_export("lite.constants.INT32").export_constant(__name__, "INT32")
_tf_export("lite.constants.INT64").export_constant(__name__, "INT64")
_tf_export("lite.constants.STRING").export_constant(__name__, "STRING")
_tf_export("lite.constants.QUANTIZED_UINT8").export_constant(
    __name__, "QUANTIZED_UINT8")
_tf_export("lite.constants.TFLITE").export_constant(__name__, "TFLITE")
_tf_export("lite.constants.GRAPHVIZ_DOT").export_constant(
    __name__, "GRAPHVIZ_DOT")

# Currently the default mode of operation is to shell to another python process
# to protect against crashes. However, it breaks some dependent targets because
# it forces us to depend on an external py_binary. The experimental API doesn't
# have that drawback.
EXPERIMENTAL_USE_TOCO_API_DIRECTLY = False


_allowed_symbols = [
    "FLOAT",
    "INT32",
    "INT64",
    "STRING",
    "QUANTIZED_UINT8",
    "COMPLEX64",
    "TENSORFLOW_GRAPHDEF",
    "TFLITE",
    "GRAPHVIZ_DOT",
    "EXPERIMENTAL_USE_TOCO_API_DIRECTLY",
]
remove_undocumented(__name__, _allowed_symbols)
