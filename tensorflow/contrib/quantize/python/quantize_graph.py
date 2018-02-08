# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""API to simulate quantization on a python graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.contrib.quantize.python import quantize
from tensorflow.python.framework import ops


def _create_graph(input_graph=None, is_training=True):
  """Rewrites input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    is_training: Whether quantizing training or eval graph.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  if is_training:
    # TODO(raghuramank): Need to make freeze_batch_norm_delay
    # a function of the batch size. For now setting this to 250 epochs
    # This corresponds to 5 million steps at a batch size of 64.
    freeze_batch_norm_delay = 5000000
  else:
    freeze_batch_norm_delay = None
  if input_graph is None:
    input_graph = ops.get_default_graph()
  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=freeze_batch_norm_delay,
        is_training=is_training)
    quantize.Quantize(input_graph, is_training=is_training)


def create_training_graph(input_graph=None):
  """Rewrites a training input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  _create_graph(input_graph=input_graph, is_training=True)


def create_eval_graph(input_graph=None):
  """Rewrites an eval input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.


  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  _create_graph(input_graph=input_graph, is_training=False)


def experimental_create_training_graph(input_graph=None):
  """Rewrites a training input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  _create_graph(input_graph=input_graph, is_training=True)


def experimental_create_eval_graph(input_graph=None):
  """Rewrites an eval input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """
  _create_graph(input_graph=input_graph, is_training=False)
