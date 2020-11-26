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
# ==============================================================================
# pylint: disable=protected-access
"""Utilities for Keras classes with v1 and v2 versions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader

# TODO(b/134426265): Switch back to single-quotes once the issue
# with copybara is fixed.
# pylint: disable=g-inconsistent-quotes
training = LazyLoader(
    "training", globals(),
    "tensorflow.python.keras.engine.training")
training_v1 = LazyLoader(
    "training_v1", globals(),
    "tensorflow.python.keras.engine.training_v1")
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
base_layer_v1 = LazyLoader(
    "base_layer_v1", globals(),
    "tensorflow.python.keras.engine.base_layer_v1")
callbacks = LazyLoader(
    "callbacks", globals(),
    "tensorflow.python.keras.callbacks")
callbacks_v1 = LazyLoader(
    "callbacks_v1", globals(),
    "tensorflow.python.keras.callbacks_v1")


# pylint: enable=g-inconsistent-quotes


class ModelVersionSelector(object):
  """Chooses between Keras v1 and v2 Model class."""

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    use_v2 = should_use_v2()
    cls = swap_class(cls, training.Model, training_v1.Model, use_v2)  # pylint: disable=self-cls-assignment
    return super(ModelVersionSelector, cls).__new__(cls)


class LayerVersionSelector(object):
  """Chooses between Keras v1 and v2 Layer class."""

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    use_v2 = should_use_v2()
    cls = swap_class(cls, base_layer.Layer, base_layer_v1.Layer, use_v2)  # pylint: disable=self-cls-assignment
    return super(LayerVersionSelector, cls).__new__(cls)


class TensorBoardVersionSelector(object):
  """Chooses between Keras v1 and v2 TensorBoard callback class."""

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    use_v2 = should_use_v2()
    start_cls = cls
    cls = swap_class(start_cls, callbacks.TensorBoard, callbacks_v1.TensorBoard,
                     use_v2)
    if start_cls == callbacks_v1.TensorBoard and cls == callbacks.TensorBoard:
      # Since the v2 class is not a subclass of the v1 class, __init__ has to
      # be called manually.
      return cls(*args, **kwargs)
    return super(TensorBoardVersionSelector, cls).__new__(cls)


def should_use_v2():
  """Determine if v1 or v2 version should be used."""
  if context.executing_eagerly():
    return True
  elif ops.executing_eagerly_outside_functions():
    # Check for a v1 `wrap_function` FuncGraph.
    # Code inside a `wrap_function` is treated like v1 code.
    graph = ops.get_default_graph()
    if (getattr(graph, "name", False) and
        graph.name.startswith("wrapped_function")):
      return False
    return True


def swap_class(cls, v2_cls, v1_cls, use_v2):
  """Swaps in v2_cls or v1_cls depending on graph mode."""
  if cls == object:
    return cls

  if cls in (v2_cls, v1_cls):
    if use_v2:
      return v2_cls
    return v1_cls

  # Recursively search superclasses to swap in the right Keras class.
  cls.__bases__ = tuple(
      swap_class(base, v2_cls, v1_cls, use_v2) for base in cls.__bases__)
  return cls


def disallow_legacy_graph(cls_name, method_name):
  if not ops.executing_eagerly_outside_functions():
    error_msg = (
        "Calling `{cls_name}.{method_name}` in graph mode is not supported "
        "when the `{cls_name}` instance was constructed with eager mode "
        "enabled. Please construct your `{cls_name}` instance in graph mode or"
        " call `{cls_name}.{method_name}` with eager mode enabled.")
    error_msg = error_msg.format(cls_name=cls_name, method_name=method_name)
    raise ValueError(error_msg)


def is_v1_layer_or_model(obj):
  return isinstance(obj, (base_layer_v1.Layer, training_v1.Model))
