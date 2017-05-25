# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.python.slim.nets import
   resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

resnet_arg_scope = resnet_utils.resnet_arg_scope


@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with variable_scope.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = layers.conv2d(
        inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    residual = resnet_utils.conv2d_same(
        residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
    residual = layers.conv2d(
        residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=None,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not. If None, the value inherited from
      the resnet_arg_scope is used. Specifying value None is deprecated.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with variable_scope.variable_scope(
      scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with arg_scope(
        [layers.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      if is_training is not None:
        bn_scope = arg_scope([layers.batch_norm], is_training=is_training)
      else:
        bn_scope = arg_scope([])
      with bn_scope:
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        if global_pool:
          # Global average pooling.
          net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        if num_classes is not None:
          net = layers.conv2d(
              net,
              num_classes, [1, 1],
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = layers_lib.softmax(
              net, scope='predictions')
        return net, end_points
resnet_v1.default_image_size = 224


def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=None,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=None,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v1_101'):
  """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=None,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v1_152'):
  """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)


def resnet_v1_200(inputs,
                  num_classes=None,
                  is_training=None,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v1_200'):
  """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=24, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)
