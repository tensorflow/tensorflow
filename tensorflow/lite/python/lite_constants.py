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

from tensorflow.compiler.mlir.lite import converter_flags_pb2 as _converter_flags_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export

FLOAT = dtypes.float32
FLOAT16 = dtypes.float16
INT32 = dtypes.int32
INT64 = dtypes.int64
STRING = dtypes.string
QUANTIZED_UINT8 = dtypes.uint8
INT8 = dtypes.int8
INT16 = dtypes.int16
COMPLEX64 = dtypes.complex64
TENSORFLOW_GRAPHDEF = _converter_flags_pb2.TENSORFLOW_GRAPHDEF
TFLITE = _converter_flags_pb2.TFLITE
GRAPHVIZ_DOT = _converter_flags_pb2.GRAPHVIZ_DOT
UNSET = _converter_flags_pb2.ConverterFlags.ModelOriginFramework.Name(
    _converter_flags_pb2.ConverterFlags.UNSET
)
TENSORFLOW = _converter_flags_pb2.ConverterFlags.ModelOriginFramework.Name(
    _converter_flags_pb2.ConverterFlags.TENSORFLOW
)
KERAS = _converter_flags_pb2.ConverterFlags.ModelOriginFramework.Name(
    _converter_flags_pb2.ConverterFlags.KERAS
)
JAX = _converter_flags_pb2.ConverterFlags.ModelOriginFramework.Name(
    _converter_flags_pb2.ConverterFlags.JAX
)
PYTORCH = _converter_flags_pb2.ConverterFlags.ModelOriginFramework.Name(
    _converter_flags_pb2.ConverterFlags.PYTORCH
)

_tf_export(v1=["lite.constants.FLOAT"]).export_constant(__name__, "FLOAT")
_tf_export(v1=["lite.constants.FLOAT16"]).export_constant(__name__, "FLOAT16")
_tf_export(v1=["lite.constants.INT32"]).export_constant(__name__, "INT32")
_tf_export(v1=["lite.constants.INT64"]).export_constant(__name__, "INT64")
_tf_export(v1=["lite.constants.STRING"]).export_constant(__name__, "STRING")
_tf_export(v1=["lite.constants.QUANTIZED_UINT8"]).export_constant(
    __name__, "QUANTIZED_UINT8")
_tf_export(v1=["lite.constants.INT8"]).export_constant(__name__, "INT8")
_tf_export(v1=["lite.constants.INT16"]).export_constant(__name__, "INT16")
_tf_export(v1=["lite.constants.TFLITE"]).export_constant(__name__, "TFLITE")
_tf_export(v1=["lite.constants.GRAPHVIZ_DOT"]).export_constant(
    __name__, "GRAPHVIZ_DOT")


_allowed_symbols = [
    "FLOAT",
    "FLOAT16",
    "INT32",
    "INT64",
    "STRING",
    "QUANTIZED_UINT8",
    "INT8",
    "INT16",
    "COMPLEX64",
    "TENSORFLOW_GRAPHDEF",
    "TFLITE",
    "GRAPHVIZ_DOT",
    "UNSET",
    "TENSORFLOW",
    "KERAS",
    "JAX",
    "PYTORCH",
]
remove_undocumented(__name__, _allowed_symbols)
