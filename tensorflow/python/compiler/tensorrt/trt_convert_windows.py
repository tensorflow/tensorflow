# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Exposes the TRT conversion for Windows platform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import platform

from tensorflow.python.util.tf_export import tf_export

if platform.system() != "Windows":
  raise RuntimeError(
      "This module is expected to be loaded only on Windows platform.")


class TrtPrecisionMode(object):
  FP32 = "FP32"
  FP16 = "FP16"
  INT8 = "INT8"


# Use a large enough number as the default max_workspace_size for TRT engines,
# so it can produce reasonable performance results with the default.
DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = 1 << 30


@tf_export("experimental.tensorrt.ConversionParams", v1=[])
class TrtConversionParams(collections.namedtuple("TrtConversionParams", [
    "rewriter_config_template", "max_workspace_size_bytes", "precision_mode",
    "minimum_segment_size", "is_dynamic_op", "maximum_cached_engines",
    "use_calibration", "max_batch_size"])):
  """Parameters that are used for TF-TRT conversion.

  Fields:
    rewriter_config_template: a template RewriterConfig proto used to create a
      TRT-enabled RewriterConfig. If None, it will use a default one.
    max_workspace_size_bytes: the maximum GPU temporary memory which the TRT
      engine can use at execution time. This corresponds to the
      'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
    precision_mode: one the strings in
      TrtPrecisionMode.supported_precision_modes().
    minimum_segment_size: the minimum number of nodes required for a subgraph
      to be replaced by TRTEngineOp.
    is_dynamic_op: whether to generate dynamic TRT ops which will build the
      TRT network and engine at run time. i.e. Since TensorRT version < 6.0
      does not support dynamic dimensions other than the batch dimension, when
      the TensorFlow graph has a non-batch dimension of dynamic size, we would
      need to enable this option. This option should be set to True in TF 2.0.
    maximum_cached_engines: max number of cached TRT engines for dynamic TRT
      ops. Created TRT engines for a dynamic dimension are cached. This is the
      maximum number of engines that can be cached. If the number of cached
      engines is already at max but none of them supports the input shapes,
      the TRTEngineOp will fall back to run the original TF subgraph that
      corresponds to the TRTEngineOp.
    use_calibration: this argument is ignored if precision_mode is not INT8.
      If set to True, a calibration graph will be created to calibrate the
      missing ranges. The calibration graph must be converted to an inference
      graph by running calibration with calibrate(). If set to False,
      quantization nodes will be expected for every tensor in the graph
      (exlcuding those which will be fused). If a range is missing, an error
      will occur. Please note that accuracy may be negatively affected if
      there is a mismatch between which tensors TRT quantizes and which
      tensors were trained with fake quantization.
    max_batch_size: max size for the input batch. This parameter is only
      effective when is_dynamic_op=False which is not supported in TF 2.0.
  """

  def __new__(cls,
              rewriter_config_template=None,
              max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
              precision_mode=TrtPrecisionMode.FP32,
              minimum_segment_size=3,
              is_dynamic_op=True,
              maximum_cached_engines=1,
              use_calibration=True,
              max_batch_size=1):
    raise NotImplementedError(
        "TensorRT integration is not available on Windows.")


@tf_export("experimental.tensorrt.Converter", v1=[])
class TrtConverterWindows(object):
  """An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

  Currently this is not available on Windows platform.
  """

  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               conversion_params=None):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      conversion_params: a TrtConversionParams instance.

    Raises:
      NotImplementedError: TRT is not supported on Windows.
    """
    raise NotImplementedError(
        "TensorRT integration is not available on Windows.")
