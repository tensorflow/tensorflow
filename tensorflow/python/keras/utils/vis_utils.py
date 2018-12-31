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
# pylint: disable=protected-access
# pylint: disable=g-import-not-at-top
"""Utilities related to model visualization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.util.tf_export import keras_export


try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None


def _check_pydot():
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
  except Exception:
    # pydot raises a generic Exception here,
    # so no specific class can be caught.
    raise ImportError('Failed to import pydot. You must install pydot'
                      ' and graphviz for `pydotprint` to work.')


def model_to_dot(model, show_shapes=False, show_layer_names=True, rankdir='TB'):
  """Convert a Keras model to dot format.

  Arguments:
      model: A Keras model instance.
      show_shapes: whether to display shape information.
      show_layer_names: whether to display layer names.
      rankdir: `rankdir` argument passed to PyDot,
          a string specifying the format of the plot:
          'TB' creates a vertical plot;
          'LR' creates a horizontal plot.

  Returns:
      A `pydot.Dot` instance representing the Keras model.
  """
  from tensorflow.python.keras.layers.wrappers import Wrapper
  from tensorflow.python.keras.models import Sequential

  _check_pydot()
  dot = pydot.Dot()
  dot.set('rankdir', rankdir)
  dot.set('concentrate', True)
  dot.set_node_defaults(shape='record')

  if isinstance(model, Sequential):
    if not model.built:
      model.build()
  layers = model._layers

  # Create graph nodes.
  for layer in layers:
    layer_id = str(id(layer))

    # Append a wrapped layer's label to node's label, if it exists.
    layer_name = layer.name
    class_name = layer.__class__.__name__
    if isinstance(layer, Wrapper):
      layer_name = '{}({})'.format(layer_name, layer.layer.name)
      child_class_name = layer.layer.__class__.__name__
      class_name = '{}({})'.format(class_name, child_class_name)

    # Create node's label.
    if show_layer_names:
      label = '{}: {}'.format(layer_name, class_name)
    else:
      label = class_name

    # Rebuild the label as a table including input/output shapes.
    if show_shapes:
      try:
        outputlabels = str(layer.output_shape)
      except AttributeError:
        outputlabels = 'multiple'
      if hasattr(layer, 'input_shape'):
        inputlabels = str(layer.input_shape)
      elif hasattr(layer, 'input_shapes'):
        inputlabels = ', '.join([str(ishape) for ishape in layer.input_shapes])
      else:
        inputlabels = 'multiple'
      label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels,
                                                     outputlabels)
    node = pydot.Node(layer_id, label=label)
    dot.add_node(node)

  # Connect nodes with edges.
  for layer in layers:
    layer_id = str(id(layer))
    for i, node in enumerate(layer._inbound_nodes):
      node_key = layer.name + '_ib-' + str(i)
      if node_key in model._network_nodes:  # pylint: disable=protected-access
        for inbound_layer in node.inbound_layers:
          inbound_layer_id = str(id(inbound_layer))
          layer_id = str(id(layer))
          dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
  return dot


@keras_export('keras.utils.plot_model')
def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB'):
  """Converts a Keras model to dot format and save to a file.

  Arguments:
      model: A Keras model instance
      to_file: File name of the plot image.
      show_shapes: whether to display shape information.
      show_layer_names: whether to display layer names.
      rankdir: `rankdir` argument passed to PyDot,
          a string specifying the format of the plot:
          'TB' creates a vertical plot;
          'LR' creates a horizontal plot.
  """
  dot = model_to_dot(model, show_shapes, show_layer_names, rankdir)
  _, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  dot.write(to_file, format=extension)
