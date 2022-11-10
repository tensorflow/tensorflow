# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Exposes the Python wrapper conversion to trt_graph."""

from tensorflow.python.compiler.tensorrt.trt_convert_common import DEFAULT_TRT_CONVERSION_PARAMS
from tensorflow.python.compiler.tensorrt.trt_convert_common import DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
from tensorflow.python.compiler.tensorrt.trt_convert_common import TrtConversionParams
from tensorflow.python.compiler.tensorrt.trt_convert_common import TrtPrecisionMode
from tensorflow.python.compiler.tensorrt.trt_convert_v1 import TrtGraphConverter
from tensorflow.python.compiler.tensorrt.trt_convert_v1 import create_inference_graph
from tensorflow.python.compiler.tensorrt.trt_convert_v1 import get_tensorrt_rewriter_config
from tensorflow.python.compiler.tensorrt.trt_convert_v2 import TrtGraphConverterV2

__all__ = [
    # Publicly accessible helper objects
    "DEFAULT_TRT_CONVERSION_PARAMS",
    "DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES",
    "TrtConversionParams",
    "TrtPrecisionMode",
    # TF v1 APIs
    "TrtGraphConverter",
    "create_inference_graph",
    "get_tensorrt_rewriter_config",
    # TF v2 APIs
    "TrtGraphConverterV2"
]
