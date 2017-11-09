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
"""Computes Receptive Field (RF) information for different models.

The receptive field (and related parameters) for the different models are
printed to stdout, and may also optionally be written to a CSV file.

For an example of usage, see rf_benchmark.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import sys

from nets import alexnet
from nets import inception
from nets import mobilenet_v1
from nets import resnet_v1
from nets import resnet_v2
from nets import vgg
from tensorflow.contrib import framework
from tensorflow.contrib import receptive_field
from tensorflow.contrib import slim
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import app

cmd_args = None

# Input node name for all architectures.
_INPUT_NODE = 'input_image'

# Variants of different network architectures.

# - resnet: different versions and sizes.
_SUPPORTED_RESNET_VARIANTS = [
    'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200',
    'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200'
]

# - inception_resnet_v2: default, and version with SAME padding.
_SUPPORTED_INCEPTIONRESNETV2_VARIANTS = [
    'inception_resnet_v2', 'inception_resnet_v2-same'
]

# - inception_v2: default, and version with no separable conv.
_SUPPORTED_INCEPTIONV2_VARIANTS = [
    'inception_v2', 'inception_v2-no-separable-conv'
]

# - inception_v3: default version.
_SUPPORTED_INCEPTIONV3_VARIANTS = ['inception_v3']

# - inception_v4: default version.
_SUPPORTED_INCEPTIONV4_VARIANTS = ['inception_v4']

# - alexnet_v2: default version.
_SUPPORTED_ALEXNETV2_VARIANTS = ['alexnet_v2']

# - vgg: vgg_a (with 11 layers) and vgg_16 (version D).
_SUPPORTED_VGG_VARIANTS = ['vgg_a', 'vgg_16']

# - mobilenet_v1: 100% and 75%.
_SUPPORTED_MOBILENETV1_VARIANTS = ['mobilenet_v1', 'mobilenet_v1_075']


def _construct_model(model_type='resnet_v1_50'):
  """Constructs model for the desired type of CNN.

  Args:
    model_type: Type of model to be used.

  Returns:
    end_points: A dictionary from components of the network to the corresponding
      activations.

  Raises:
    ValueError: If the model_type is not supported.
  """
  # Placeholder input.
  images = array_ops.placeholder(
      dtypes.float32, shape=(1, None, None, 3), name=_INPUT_NODE)

  # Construct model.
  if model_type == 'inception_resnet_v2':
    _, end_points = inception.inception_resnet_v2_base(images)
  elif model_type == 'inception_resnet_v2-same':
    _, end_points = inception.inception_resnet_v2_base(
        images, align_feature_maps=True)
  elif model_type == 'inception_v2':
    _, end_points = inception.inception_v2_base(images)
  elif model_type == 'inception_v2-no-separable-conv':
    _, end_points = inception.inception_v2_base(
        images, use_separable_conv=False)
  elif model_type == 'inception_v3':
    _, end_points = inception.inception_v3_base(images)
  elif model_type == 'inception_v4':
    _, end_points = inception.inception_v4_base(images)
  elif model_type == 'alexnet_v2':
    _, end_points = alexnet.alexnet_v2(images)
  elif model_type == 'vgg_a':
    _, end_points = vgg.vgg_a(images)
  elif model_type == 'vgg_16':
    _, end_points = vgg.vgg_16(images)
  elif model_type == 'mobilenet_v1':
    _, end_points = mobilenet_v1.mobilenet_v1_base(images)
  elif model_type == 'mobilenet_v1_075':
    _, end_points = mobilenet_v1.mobilenet_v1_base(
        images, depth_multiplier=0.75)
  elif model_type == 'resnet_v1_50':
    _, end_points = resnet_v1.resnet_v1_50(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v1_101':
    _, end_points = resnet_v1.resnet_v1_101(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v1_152':
    _, end_points = resnet_v1.resnet_v1_152(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v1_200':
    _, end_points = resnet_v1.resnet_v1_200(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v2_50':
    _, end_points = resnet_v2.resnet_v2_50(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v2_101':
    _, end_points = resnet_v2.resnet_v2_101(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v2_152':
    _, end_points = resnet_v2.resnet_v2_152(
        images, num_classes=None, is_training=False, global_pool=False)
  elif model_type == 'resnet_v2_200':
    _, end_points = resnet_v2.resnet_v2_200(
        images, num_classes=None, is_training=False, global_pool=False)
  else:
    raise ValueError('Unsupported model_type %s.' % model_type)

  return end_points


def _get_desired_end_point_keys(model_type='resnet_v1_50'):
  """Gets list of desired end point keys for a type of CNN.

  Args:
    model_type: Type of model to be used.

  Returns:
    desired_end_point_types: A list containing the desired end-points.

  Raises:
    ValueError: If the model_type is not supported.
  """
  if model_type in _SUPPORTED_RESNET_VARIANTS:
    blocks = ['block1', 'block2', 'block3', 'block4']
    desired_end_point_keys = ['%s/%s' % (model_type, i) for i in blocks]
  elif model_type in _SUPPORTED_INCEPTIONRESNETV2_VARIANTS:
    desired_end_point_keys = [
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'MaxPool_3a_3x3',
        'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3', 'Mixed_5b',
        'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1'
    ]
  elif model_type in _SUPPORTED_INCEPTIONV2_VARIANTS:
    desired_end_point_keys = [
        'Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3',
        'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a', 'Mixed_4b',
        'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c'
    ]
  elif model_type in _SUPPORTED_INCEPTIONV3_VARIANTS:
    desired_end_point_keys = [
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'MaxPool_3a_3x3',
        'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3', 'Mixed_5b',
        'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
        'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'
    ]
  elif model_type in _SUPPORTED_INCEPTIONV4_VARIANTS:
    desired_end_point_keys = [
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Mixed_3a',
        'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_5e',
        'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e', 'Mixed_6f',
        'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c', 'Mixed_7d'
    ]
  elif model_type in _SUPPORTED_ALEXNETV2_VARIANTS:
    ep = ['conv1', 'pool1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool5']
    desired_end_point_keys = ['%s/%s' % (model_type, i) for i in ep]
  elif model_type in _SUPPORTED_VGG_VARIANTS:
    ep = [
        'conv1/conv1_1', 'pool1', 'conv2/conv2_1', 'pool2', 'conv3/conv3_1',
        'conv3/conv3_2', 'pool3', 'conv4/conv4_1', 'conv4/conv4_2', 'pool4',
        'conv5/conv5_1', 'conv5/conv5_2', 'pool5'
    ]
    desired_end_point_keys = ['%s/%s' % (model_type, i) for i in ep]
  elif model_type in _SUPPORTED_MOBILENETV1_VARIANTS:
    desired_end_point_keys = [
        'Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
        'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5_pointwise',
        'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
        'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
        'Conv2d_12_pointwise', 'Conv2d_13_pointwise'
    ]
  else:
    raise ValueError('Unsupported model_type %s.' % model_type)

  return desired_end_point_keys


def _model_graph_def(model_type='resnet_v1_50', arg_sc=None):
  """Constructs a model graph, returning GraphDef and end-points.

  Args:
    model_type: Type of model to be used.
    arg_sc: Optional arg scope to use in constructing the graph.

  Returns:
    graph_def: GraphDef of constructed graph.
    end_points: A dictionary from components of the network to the corresponding
      activations.
  """
  if arg_sc is None:
    arg_sc = {}
  g = ops.Graph()
  with g.as_default():
    with framework.arg_scope(arg_sc):
      end_points = _construct_model(model_type)

  return g.as_graph_def(), end_points


def _model_rf(graphdef,
              end_points,
              desired_end_point_keys,
              model_type='resnet_v1_50',
              csv_writer=None):
  """Computes receptive field information for a given CNN model.

  The information will be printed to stdout. If the RF parameters are the same
  for the horizontal and vertical directions, it will be printed only once.
  Otherwise, they are printed once for the horizontal and once for the vertical
  directions.

  Args:
    graphdef: GraphDef of given model.
    end_points: A dictionary from components of the model to the corresponding
      activations.
    desired_end_point_keys: List of desired end points for which receptive field
      information will be computed.
    model_type: Type of model to be used, used only for printing purposes.
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for desired_end_point_key in desired_end_point_keys:
    print('- %s:' % desired_end_point_key)
    output_node_with_colon = end_points[desired_end_point_key].name
    pos = output_node_with_colon.rfind(':')
    output_node = output_node_with_colon[:pos]
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y
    ) = receptive_field.compute_receptive_field_from_graph_def(
        graphdef, _INPUT_NODE, output_node)
    # If values are the same in horizontal/vertical directions, just report one
    # of them. Otherwise, report both.
    if (receptive_field_x == receptive_field_y) and (
        effective_stride_x == effective_stride_y) and (
            effective_padding_x == effective_padding_y):
      print('Receptive field size = %5s, effective stride = %5s, effective '
            'padding = %5s' % (str(receptive_field_x), str(effective_stride_x),
                               str(effective_padding_x)))
    else:
      print('Receptive field size: horizontal = %5s, vertical = %5s. '
            'Effective stride: horizontal = %5s, vertical = %5s. Effective '
            'padding: horizontal = %5s, vertical = %5s' %
            (str(receptive_field_x), str(receptive_field_y),
             str(effective_stride_x), str(effective_stride_y),
             str(effective_padding_x), str(effective_padding_y)))
    if csv_writer is not None:
      csv_writer.writerow({
          'CNN': model_type,
          'end_point': desired_end_point_key,
          'RF size hor': str(receptive_field_x),
          'RF size ver': str(receptive_field_y),
          'effective stride hor': str(effective_stride_x),
          'effective stride ver': str(effective_stride_y),
          'effective padding hor': str(effective_padding_x),
          'effective padding ver': str(effective_padding_y)
      })


def _process_model_rf(model_type='resnet_v1_50', csv_writer=None, arg_sc=None):
  """Contructs model graph and desired end-points, and compute RF.

  The computed RF parameters are printed to stdout by the _model_rf function.

  Args:
    model_type: Type of model to be used.
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
    arg_sc: Optional arg scope to use in constructing the graph.

  """
  print('********************%s' % model_type)
  graphdef, end_points = _model_graph_def(model_type, arg_sc)
  desired_end_point_keys = _get_desired_end_point_keys(model_type)
  _model_rf(graphdef, end_points, desired_end_point_keys, model_type,
            csv_writer)


def _resnet_rf(csv_writer=None):
  """Computes RF and associated parameters for resnet models.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_RESNET_VARIANTS:
    arg_sc = resnet_v1.resnet_arg_scope()
    _process_model_rf(model_type, csv_writer, arg_sc)


def _inception_resnet_v2_rf(csv_writer=None):
  """Computes RF and associated parameters for the inception_resnet_v2 model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_INCEPTIONRESNETV2_VARIANTS:
    _process_model_rf(model_type, csv_writer)


def _inception_v2_rf(csv_writer=None):
  """Computes RF and associated parameters for the inception_v2 model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_INCEPTIONV2_VARIANTS:
    _process_model_rf(model_type, csv_writer)


def _inception_v3_rf(csv_writer=None):
  """Computes RF and associated parameters for the inception_v3 model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_INCEPTIONV3_VARIANTS:
    _process_model_rf(model_type, csv_writer)


def _inception_v4_rf(csv_writer=None):
  """Computes RF and associated parameters for the inception_v4 model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_INCEPTIONV4_VARIANTS:
    _process_model_rf(model_type, csv_writer)


def _alexnet_v2_rf(csv_writer=None):
  """Computes RF and associated parameters for the alexnet_v2 model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_ALEXNETV2_VARIANTS:
    _process_model_rf(model_type, csv_writer)


def _vgg_rf(csv_writer=None):
  """Computes RF and associated parameters for the vgg model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_VGG_VARIANTS:
    _process_model_rf(model_type, csv_writer)


def _mobilenet_v1_rf(csv_writer=None):
  """Computes RF and associated parameters for the mobilenet_v1 model.

  The computed values are written to stdout.

  Args:
    csv_writer: A CSV writer for RF parameters, which is used if it is not None.
  """
  for model_type in _SUPPORTED_MOBILENETV1_VARIANTS:
    with slim.arg_scope(
        [slim.batch_norm, slim.dropout], is_training=False) as arg_sc:
      _process_model_rf(model_type, csv_writer, arg_sc)


def main(unused_argv):
  # Configure CSV file which will be written, if desired.
  if cmd_args.csv_path:
    csv_file = open(cmd_args.csv_path, 'w')
    field_names = [
        'CNN', 'end_point', 'RF size hor', 'RF size ver',
        'effective stride hor', 'effective stride ver', 'effective padding hor',
        'effective padding ver'
    ]
    rf_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    rf_writer.writeheader()
  else:
    rf_writer = None

  # Compute RF parameters for each network architecture.
  _alexnet_v2_rf(rf_writer)
  _vgg_rf(rf_writer)
  _inception_v2_rf(rf_writer)
  _inception_v3_rf(rf_writer)
  _inception_v4_rf(rf_writer)
  _inception_resnet_v2_rf(rf_writer)
  _mobilenet_v1_rf(rf_writer)
  _resnet_rf(rf_writer)

  # Close CSV file, if it was opened.
  if cmd_args.csv_path:
    csv_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--csv_path',
      type=str,
      default='',
      help="""\
      Path to CSV file that will be written with RF parameters.If empty, no
      file will be written.\
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
