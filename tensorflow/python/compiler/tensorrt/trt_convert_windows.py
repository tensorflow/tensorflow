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

import platform

from tensorflow.python.util.tf_export import tf_export

if platform.system() != "Windows":
  raise RuntimeError(
      "This module is expected to be loaded only on Windows platform.")


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
