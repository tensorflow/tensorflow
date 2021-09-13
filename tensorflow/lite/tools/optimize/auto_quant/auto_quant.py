# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Conversion modules for Auto Selective Quantization."""

import copy
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from tensorflow.lite.tools.optimize.debugging.python import debugger
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info


def _quant_single_layer(quant_node_name: str, node_list: List[str],
                        converter: tf.lite.TFLiteConverter) -> bytes:
  """Build a single layer quantized model.

  Args:
    quant_node_name: Name of a op to quantize.
    node_list: List of every op names.
    converter: TFLiteConverter to quantize a model.

  Returns:
    TFLite model with a single layer quantized.
  """
  for i, node_name in enumerate(node_list):
    if node_name == quant_node_name:
      node_list = node_list[:i] + node_list[i + 1:]
      break

  quant_debug_option = debugger.QuantizationDebugOptions()
  quant_debug_option.denylisted_nodes = node_list

  quant_debugger = debugger.QuantizationDebugger(
      converter=converter, debug_options=quant_debug_option)

  return quant_debugger.get_nondebug_quantized_model()


def quant_single_layers(model_path: str,
                        data_gen: Callable[[], Iterable[Sequence[np.ndarray]]],
                        quant_limit_n: int = -1) -> List[Tuple[bytes, str]]:
  """Creates a list of models, where each model has a single quantized layer and every layer is quantized exactly once.

  Args:
    model_path: Path of the SavedModel to quantize.
    data_gen: Representative dataset fetcher for quantization.
    quant_limit_n: The maximum number of single layers to quantize, -1 if
      unlimited.

  Returns:
    A list of TFLite models with where single layer is quantized each, paired
    with the name of quantized layer's name.
  """

  converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = data_gen

  model, _ = parse_saved_model_with_debug_info(model_path)

  node_names = []
  for meta_graph in model.meta_graphs:
    node_names.extend(node.name for node in meta_graph.graph_def.node)

  ret = []
  quant_cnt = 0
  for node_name in node_names:
    ret.append((_quant_single_layer(
        quant_node_name=node_name,
        node_list=copy.deepcopy(node_names),
        converter=converter), node_name))
    quant_cnt += 1
    if quant_limit_n != -1 and quant_cnt >= quant_limit_n:
      break

  return ret
