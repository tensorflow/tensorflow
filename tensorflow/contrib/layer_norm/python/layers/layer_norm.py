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

# pylint: disable=g-short-docstring-punctuation
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layer_norm.python.ops.layer_norm_fused_op import layer_norm_fused_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops

__all__ = ['layer_norm_fused']


@add_arg_scope
def layer_norm_fused(inputs,
                     center=True,
                     scale=True,
                     activation_fn=None,
                     reuse=None,
                     variables_collections=None,
                     outputs_collections=None,
                     trainable=True,
                     epsilon=1E-12,
                     scope=None):
    """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.
      "Layer Normalization"
      Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    Faster and more efficient implementation of layer normalization,
    only works on GPU.  

    Can be used as a normalizer function for any shape of input larger than 2-D,
    only accepts dtypes of float and double.

    IMPORTANT: This layer is not the same as layer_norm if the rank of the input tensor
    is larger than 2, since layer_norm normalizes over all dimensions except the first,
    while this layer normalizes along the last dimension. You can reshape the input to 2-D
    before passing it into this layer, to achieve similar effect as layer_norm, but note
    that the current implementation of layer_norm_fused kernel has a size limit of 5120 
    for last dimension.

    Args:
      inputs: a tensor with 2 or more dimensions. The normalization
              occurs along the LAST DIMENSION.
      center: If True, subtract `beta`. If False, `beta` is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      activation_fn: activation function, default set to None to skip it and
        maintain a linear activation.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional collections for the variables.
      outputs_collections: collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      epsilon: small value added to prevent NaN outputs.
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    Raises:
      ValueError: if rank or last dimension of `inputs` is undefined.
    """
    with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=init_ops.zeros_initializer(),
                                            collections=beta_collections,
                                            trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma = variables.model_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer(),
                collections=gamma_collections,
                trainable=trainable)

        outputs = layer_norm_fused_op(inputs, gamma=gamma, beta=beta,
                                      epsilon=epsilon)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope,
                                           outputs)
