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
"""Registry for layers and their parameters/variables.

This represents the collection of all layers in the approximate Fisher
information matrix to which a particular FisherBlock may belong. That is, we
might have several layer collections for one TF graph (if we have multiple K-FAC
optimizers being used, for example.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial

import math
import six

from tensorflow.contrib.kfac.python.ops import fisher_blocks as fb
from tensorflow.contrib.kfac.python.ops import loss_functions as lf
from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# Names for various approximations that can be requested for Fisher blocks.
APPROX_KRONECKER_NAME = "kron"
APPROX_DIAGONAL_NAME = "diagonal"
APPROX_FULL_NAME = "full"

_GENERIC_APPROX_TO_BLOCK_TYPES = {
    APPROX_FULL_NAME: fb.FullFB,
    APPROX_DIAGONAL_NAME: fb.NaiveDiagonalFB,
}

_FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: fb.FullyConnectedKFACBasicFB,
    APPROX_DIAGONAL_NAME: fb.FullyConnectedDiagonalFB,
}

_CONV2D_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: fb.ConvKFCBasicFB,
    APPROX_DIAGONAL_NAME: fb.ConvDiagonalFB,
}

_EMBEDDING_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: fb.EmbeddingKFACFB
}

APPROX_KRONECKER_INDEP_NAME = "kron_indep"
APPROX_KRONECKER_SERIES_1_NAME = "kron_series_1"
APPROX_KRONECKER_SERIES_2_NAME = "kron_series_2"

_FULLY_CONNECTED_MULTI_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_INDEP_NAME: fb.FullyConnectedMultiIndepFB,
    APPROX_KRONECKER_SERIES_1_NAME: partial(fb.FullyConnectedSeriesFB,
                                            option=1),
    APPROX_KRONECKER_SERIES_2_NAME: partial(fb.FullyConnectedSeriesFB,
                                            option=2)
}

_CONV2D_MULTI_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_INDEP_NAME: fb.ConvKFCBasicMultiIndepFB
}

_EMBEDDING_MULTI_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_INDEP_NAME: fb.EmbeddingKFACMultiIndepFB
}

# Possible value for `reuse` keyword argument. Sets `reuse` to
# tf.get_variable_scope().reuse.
VARIABLE_SCOPE = "VARIABLE_SCOPE"

_DEFAULT_LAYER_COLLECTION = None


def get_default_layer_collection():
  """Get default LayerCollection."""
  if _DEFAULT_LAYER_COLLECTION is None:
    raise ValueError(
        "Attempted to retrieve default LayerCollection when none is set. Use "
        "LayerCollection.as_default().")

  return _DEFAULT_LAYER_COLLECTION


def set_default_layer_collection(layer_collection):
  global _DEFAULT_LAYER_COLLECTION

  if _DEFAULT_LAYER_COLLECTION is not None and layer_collection is not None:
    raise ValueError("Default LayerCollection is already set.")

  _DEFAULT_LAYER_COLLECTION = layer_collection


class LayerParametersDict(OrderedDict):
  """An OrderedDict where keys are Tensors or tuples of Tensors.

  Ensures that no Tensor is associated with two different keys.
  """

  def __init__(self, *args, **kwargs):
    self._tensors = set()
    super(LayerParametersDict, self).__init__(*args, **kwargs)

  def __setitem__(self, key, value):
    key = self._canonicalize_key(key)
    tensors = key if isinstance(key, (tuple, list)) else (key,)
    key_collisions = self._tensors.intersection(tensors)
    if key_collisions:
      raise ValueError("Key(s) already present: {}".format(key_collisions))
    self._tensors.update(tensors)
    super(LayerParametersDict, self).__setitem__(key, value)

  def __delitem__(self, key):
    key = self._canonicalize_key(key)
    self._tensors.remove(key)
    super(LayerParametersDict, self).__delitem__(key)

  def __getitem__(self, key):
    key = self._canonicalize_key(key)
    return super(LayerParametersDict, self).__getitem__(key)

  def __contains__(self, key):
    key = self._canonicalize_key(key)
    return super(LayerParametersDict, self).__contains__(key)

  def _canonicalize_key(self, key):
    if isinstance(key, (list, tuple)):
      return tuple(key)
    return key


# TODO(b/68034464): add capability for LayerCollection to be "finalized"
# and do this when it gets used by FisherEstimator / KfacOptimizer.


class LayerCollection(object):
  """Registry of information about layers and losses.

  Note that you need to create a new one of these for each MatrixEstimator or
  KfacOptimizer.

  Attributes:
    fisher_blocks: a LayersParamsDict (subclass of OrderedDict) mapping layer
        parameters (Tensors or tuples of Tensors) to FisherBlock instances.
    fisher_factors: an OrderedDict mapping tuples to FisherFactor instances.
    losses: a list of LossFunction objects. The loss to be optimized is their
        sum.
    loss_colocation_ops: ops to colocate loss function evaluations with.  These
        will typically be the inputs to the losses.
  """

  def __init__(self,
               graph=None,
               name="LayerCollection"):
    self.fisher_blocks = LayerParametersDict()
    self.fisher_factors = OrderedDict()
    self._linked_parameters = dict(
    )  # dict mapping sets of variables to optionally specified approximations.
    self._graph = graph or ops.get_default_graph()
    self._loss_dict = {}  # {str: LossFunction}
    self._subgraph = None
    self._default_generic_approximation = APPROX_FULL_NAME
    self._default_embedding_approximation = APPROX_KRONECKER_NAME
    self._default_fully_connected_approximation = APPROX_KRONECKER_NAME
    self._default_conv2d_approximation = APPROX_KRONECKER_NAME
    self._default_fully_connected_multi_approximation = (
        APPROX_KRONECKER_INDEP_NAME)
    self._default_conv2d_multi_approximation = (
        APPROX_KRONECKER_INDEP_NAME)
    self._default_embedding_multi_approximation = APPROX_KRONECKER_INDEP_NAME
    self.loss_colocation_ops = {}
    self._vars_to_uses = defaultdict(lambda: 0)

    with variable_scope.variable_scope(None, default_name=name) as scope:
      self._var_scope = scope.name

  @property
  def losses(self):
    """Tuple of LossFunction objects registered with this LayerCollection."""
    return nest.flatten(self.towers_by_loss)

  @property
  def towers_by_loss(self):
    """Tuple across losses of LossFunction objects registered to each tower."""
    return tuple(tuple(lst) for lst in self._loss_dict.values())

  @property
  def registered_variables(self):
    """A tuple of all of the variables currently registered."""
    tuple_of_tuples = (utils.ensure_sequence(key) for key, block
                       in six.iteritems(self.fisher_blocks))
    flat_tuple = tuple(item for tuple_ in tuple_of_tuples for item in tuple_)
    return flat_tuple

  @property
  def linked_parameters(self):
    """Groups of parameters with an optionally specified approximation.

    Linked parameters can be added using `define_linked_parameters`.
    If an approximation is specified, then this approximation will be used
    when registering a layer with exactly these parameters, unless an
    approximation is specified when calling the registration function.

    Returns:
      A `dict` mapping tuples of parameters to an optional string.
    """
    return self._linked_parameters

  @property
  def default_embedding_approximation(self):
    return self._default_embedding_approximation

  def set_default_embedding_approximation(self, value):
    if value != APPROX_KRONECKER_NAME:
      raise ValueError(
          "{} is not a valid approximation for embedding variables.".format(
              value))
    self._default_embedding_approximation = value

  @property
  def default_generic_approximation(self):
    return self._default_generic_approximation

  def set_default_generic_approximation(self, value):
    if value not in _GENERIC_APPROX_TO_BLOCK_TYPES:
      raise ValueError(
          "{} is not a valid approximation for generic variables.".format(
              value))
    self._default_generic_approximation = value

  @property
  def default_fully_connected_approximation(self):
    return self._default_fully_connected_approximation

  def set_default_fully_connected_approximation(self, value):
    if value not in _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES:
      raise ValueError(
          "{} is not a valid approximation for fully connected layers.".format(
              value))
    self._default_fully_connected_approximation = value

  @property
  def default_conv2d_approximation(self):
    return self._default_conv2d_approximation

  def set_default_conv2d_approximation(self, value):
    if value not in _CONV2D_APPROX_TO_BLOCK_TYPES:
      raise ValueError(
          "{} is not a valid approximation for 2d convolutional layers.".format(
              value))
    self._default_conv2d_approximation = value

  @property
  def default_fully_connected_multi_approximation(self):
    return self._default_fully_connected_multi_approximation

  def set_default_fully_connected_multi_approximation(self, value):
    if value not in _FULLY_CONNECTED_MULTI_APPROX_TO_BLOCK_TYPES:
      raise ValueError("{} is not a valid approximation for a fully-connected "
                       "multi layer.".format(value))
    self._default_fully_connected_multi_approximation = value

  @property
  def default_conv2d_multi_approximation(self):
    return self._default_conv2d_multi_approximation

  @property
  def default_embedding_multi_approximation(self):
    return self._default_embedding_multi_approximation

  def register_block(self, layer_key, fisher_block, reuse=VARIABLE_SCOPE):
    """Validates and registers the layer_key associated with the fisher_block.

    Args:
      layer_key: A variable or tuple of variables. The key to check for in
          existing registrations and to register if valid.
      fisher_block: The associated `FisherBlock`.
      reuse: Method to use for inserting new `FisherBlock's. One of True, False,
        or `VARIABLE_SCOPE`.

    Raises:
      ValueError: If `layer_key` was already registered and reuse is `False`,
        if `layer_key` was registered with a different block type, or if
        `layer_key` shares any variables with but is not equal to a previously
        registered key.
      KeyError: If `reuse` is `True` but `layer_key` was not previously
        registered.

    Returns:
      The `FisherBlock` registered under `layer_key`. If `layer_key` was already
      registered, this will be the previously registered `FisherBlock`.
    """
    if reuse is VARIABLE_SCOPE:
      reuse = variable_scope.get_variable_scope().reuse

    if reuse is True or (reuse is variable_scope.AUTO_REUSE and
                         layer_key in self.fisher_blocks):
      result = self.fisher_blocks[layer_key]
      if type(result) != type(fisher_block):  # pylint: disable=unidiomatic-typecheck
        raise ValueError(
            "Attempted to register FisherBlock of type %s when existing "
            "FisherBlock has type %s." % (type(fisher_block), type(result)))
      return result
    if reuse is False and layer_key in self.fisher_blocks:
      raise ValueError("FisherBlock for %s is already in LayerCollection." %
                       (layer_key,))

    # Insert fisher_block into self.fisher_blocks.
    if layer_key in self.fisher_blocks:
      raise ValueError("Duplicate registration: {}".format(layer_key))
    # Raise an error if any variable in layer_key has been registered in any
    # other blocks.
    variable_to_block = {
        var: (params, block)
        for (params, block) in self.fisher_blocks.items()
        for var in utils.ensure_sequence(params)
    }
    for variable in utils.ensure_sequence(layer_key):
      if variable in variable_to_block:
        prev_key, prev_block = variable_to_block[variable]
        raise ValueError(
            "Attempted to register layer_key {} with block {}, but variable {}"
            " was already registered in key {} with block {}.".format(
                layer_key, fisher_block, variable, prev_key, prev_block))
    self.fisher_blocks[layer_key] = fisher_block
    return fisher_block

  def register_loss_function(self,
                             loss,
                             colocation_op,
                             base_name,
                             name=None,
                             reuse=VARIABLE_SCOPE):
    """Registers a LossFunction object.

    Args:
      loss: The LossFunction object.
      colocation_op: The op to colocate the loss function's computations with.
      base_name: The name to derive a new unique name from is the name argument
        is None.
      name: (OPTIONAL) str or None. Unique name for this loss function. If None,
        a new name is generated. (Default: None)
      reuse: (OPTIONAL) bool or str.  If True, adds `loss` as an additional
        tower for the existing loss function.

    Raises:
      ValueError: If reuse == True and name == None.
      ValueError: If reuse == True and seed != None.
      KeyError: If reuse == True and no existing LossFunction with `name` found.
      KeyError: If reuse == False and existing LossFunction with `name` found.
    """

    name = name or self._graph.unique_name(base_name)

    if reuse == VARIABLE_SCOPE:
      reuse = variable_scope.get_variable_scope().reuse

    if reuse:
      if name is None:
        raise ValueError(
            "If reuse is enabled, loss function's name must be set.")

      loss_list = self._loss_dict.get(name, None)

      if loss_list is None:
        raise KeyError(
            "Unable to find loss function named {}. Register a new loss "
            "function with reuse=False.".format(name))
    else:
      if name in self._loss_dict:
        raise KeyError(
            "Loss function named {} already exists. Set reuse=True to append "
            "another tower.".format(name))

      loss_list = []
      self._loss_dict[name] = loss_list

    loss_list.append(loss)
    self.loss_colocation_ops[loss] = colocation_op

  def _get_use_count_map(self):
    """Returns a dict mapping variables to their number of registrations."""
    return self._vars_to_uses

  def _add_uses(self, params, uses):
    """Register additional uses by params in the graph.

    Args:
      params: Variable or tuple of Variables. Parameters for a layer.
      uses: int or float. Number of additional uses for these parameters.
    """
    params = params if isinstance(params, (tuple, list)) else (params,)
    for var in params:
      self._vars_to_uses[var] += uses

  def check_registration(self, variables):
    """Checks that all variable uses have been registered properly.

    Args:
      variables: List of variables.

    Raises:
      ValueError: If any registered variables are not included in the list.
      ValueError: If any variable in the list is not registered.
      ValueError: If any variable in the list is registered with the wrong
          number of "uses" in the subgraph recorded (vs the number of times that
          variable is actually used in the subgraph).
    """
    # Note that overlapping parameters (i.e. those that share variables) will
    # be caught by layer_collection.LayerParametersDict during registration.

    reg_use_map = self._get_use_count_map()

    error_messages = []

    for var in variables:
      total_uses = self.subgraph.variable_uses(var)
      reg_uses = reg_use_map[var]

      if reg_uses == 0:
        error_messages.append("Variable {} not registered.".format(var))
      elif (not math.isinf(reg_uses)) and reg_uses != total_uses:
        error_messages.append(
            "Variable {} registered with wrong number of uses ({} "
            "registrations vs {} uses).".format(var, reg_uses, total_uses))

    num_get_vars = len(reg_use_map)

    if num_get_vars > len(variables):
      error_messages.append("{} registered variables were not included in list."
                            .format(num_get_vars - len(variables)))

    if error_messages:
      error_messages = [
          "Found the following errors with variable registration:"
      ] + error_messages
      raise ValueError("\n\t".join(error_messages))

  def get_blocks(self):
    return self.fisher_blocks.values()

  def get_factors(self):
    return self.fisher_factors.values()

  @property
  def graph(self):
    return self._graph

  @property
  def subgraph(self):
    return self._subgraph

  def define_linked_parameters(self, params, approximation=None):
    """Identify a set of parameters that should be grouped together.

    During automatic graph scanning, any matches containing variables that have
    been identified as part of a linked group will be filtered out unless
    the match parameters are exactly equal to the ones specified in the linked
    group.

    Args:
      params: A variable, or a tuple or list of variables. The variables
        to be linked.
      approximation: Optional string specifying the type of approximation to use
        for these variables. If unspecified, this layer collection's default
        approximation for the layer type will be used.

    Raises:
      ValueError: If the parameters were already registered in a layer or
        identified as part of an incompatible group.
    """
    params = frozenset(utils.ensure_sequence(params))

    # Check if any of the variables in `params` is already in
    # 'self.fisher_blocks.keys()`.
    for registered_params, fisher_block in self.fisher_blocks.items():
      registered_params_set = set(utils.ensure_sequence(registered_params))
      for variable in params:
        if (variable in registered_params_set and
            params != registered_params_set):
          raise ValueError(
              "Can`t link parameters {}, variable {} was already registered in "
              "group {} with layer {}".format(params, variable,
                                              registered_params, fisher_block))

    # Check if any of the variables in `params` is already in
    # 'self.linked_parameters`.
    for variable in params:
      for other_linked_params in self.linked_parameters:
        if variable in other_linked_params:
          raise ValueError("Can`t link parameters {}, variable {} was already "
                           "linked in group {}.".format(params, variable,
                                                        other_linked_params))
    self._linked_parameters[params] = approximation

  def create_subgraph(self):
    if not self.losses:
      raise ValueError("Must have at least one registered loss.")
    inputs_to_losses = nest.flatten(tuple(loss.inputs for loss in self.losses))
    self._subgraph = utils.SubGraph(inputs_to_losses)

  def eval_losses(self):
    """Return evaluated losses (colocated with inputs to losses)."""
    evals = []
    for loss in self.losses:
      with ops.colocate_with(self.loss_colocation_ops[loss]):
        evals.append(loss.evaluate())
    return evals

  def eval_losses_on_samples(self):
    """Return losses evaluated on samples (colocated with inputs to losses)."""
    evals = []
    for loss in self.losses:
      with ops.colocate_with(self.loss_colocation_ops[loss]):
        evals.append(loss.evaluate_on_sample())
    return evals

  def total_loss(self):
    return math_ops.add_n(self.eval_losses())

  def total_sampled_loss(self):
    return math_ops.add_n(self.eval_losses_on_samples())

  def _get_linked_approx(self, params):
    """If params were linked, return their specified approximation."""
    params_set = frozenset(utils.ensure_sequence(params))
    if params_set in self.linked_parameters:
      return self.linked_parameters[params_set]
    else:
      return None

  def _get_block_type(self, params, approx, default, approx_to_type):
    if approx is None:
      approx = self._get_linked_approx(params)
      if approx is None:
        approx = default

    if approx not in approx_to_type:
      raise ValueError("Bad value {} for approx.".format(approx))

    return approx_to_type[approx], approx

  def register_embedding(self,
                         params,
                         inputs,
                         outputs,
                         approx=None,
                         reuse=VARIABLE_SCOPE):
    """Registers an embedding layer.

    Args:
      params: Embedding matrix of shape [vocab_size, embedding_size].
      inputs: Tensor of shape [batch_size, input_size] and dtype int32. Indices
        into embedding matrix.
      outputs: Tensor of shape [batch_size, embedding_size]. Outputs
        produced by layer.
      approx: str or None. If not None must be "kron".  The Fisher
        approximation to use. If None the default value is used. (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    block_type, approx = self._get_block_type(
        params, approx, self.default_embedding_approximation,
        _EMBEDDING_APPROX_TO_BLOCK_TYPES)

    if isinstance(params, (tuple, list)):
      raise ValueError("Bias not supported.")
    vocab_size = int(params.shape[0])
    block = self.register_block(
        params, block_type(self, vocab_size), reuse=reuse)
    block.register_additional_tower(inputs, outputs)

    self._add_uses(params, 1)

  def register_fully_connected(self,
                               params,
                               inputs,
                               outputs,
                               approx=None,
                               reuse=VARIABLE_SCOPE):
    """Registers a fully connnected layer.

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [input_size, output_size].
        Bias should have shape [output_size].
      inputs: Tensor of shape [batch_size, input_size]. Inputs to layer.
      outputs: Tensor of shape [batch_size, output_size]. Outputs
        produced by layer.
      approx: str or None. If not None must be one of "kron" or "diagonal".
        The Fisher approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """

    block_type, approx = self._get_block_type(
        params, approx, self.default_fully_connected_approximation,
        _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES)

    has_bias = isinstance(params, (tuple, list))
    block = self.register_block(params, block_type(self, has_bias=has_bias),
                                reuse=reuse)
    block.register_additional_tower(inputs, outputs)

    self._add_uses(params, 1)

  def register_conv2d(self,
                      params,
                      strides,
                      padding,
                      inputs,
                      outputs,
                      data_format=None,
                      dilations=None,
                      approx=None,
                      reuse=VARIABLE_SCOPE):
    """Registers a call to tf.nn.conv2d().

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [kernel_height,
        kernel_width, in_channels, out_channels].  Bias should have shape
        [out_channels].
      strides: List of 4 ints. Strides for convolution kernel.
      padding: string. see tf.nn.conv2d for valid values.
      inputs: Tensor of shape [batch_size, height, width, in_channels]. Inputs
        to layer.
      outputs: Tensor of shape [batch_size, height, width, out_channels].
        Output produced by layer.
      data_format: str or None. Format of data.
      dilations: List of 4 ints. Dilations along each dimension.
      approx: str or None. If not None must be one of "kron" or "diagonal".
        The Fisher approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """

    block_type, approx = self._get_block_type(
        params, approx, self.default_conv2d_approximation,
        _CONV2D_APPROX_TO_BLOCK_TYPES)

    # It feels bad to pass in configuration that has to do with the internal
    # implementation.  And then we can`t use the same constructor for both
    # anymore and are thus forced to use this ugly if-statement.
    # TODO(b/74793309): Clean this up?
    if approx == APPROX_KRONECKER_NAME:
      block = self.register_block(
          params,
          block_type(
              layer_collection=self,
              params=params,
              padding=padding,
              strides=strides,
              data_format=data_format,
              dilation_rate=dilations,
              extract_patches_fn="extract_image_patches"),
          reuse=reuse)
    elif approx == APPROX_DIAGONAL_NAME:
      assert strides[0] == strides[-1] == 1
      block = self.register_block(
          params,
          block_type(
              layer_collection=self,
              params=params,
              padding=padding,
              strides=strides,
              dilations=dilations,
              data_format=data_format),
          reuse=reuse)
    else:
      raise NotImplementedError(approx)

    block.register_additional_tower(inputs, outputs)

    self._add_uses(params, 1)

  def register_convolution(self,
                           params,
                           inputs,
                           outputs,
                           padding,
                           strides=None,
                           dilation_rate=None,
                           data_format=None,
                           approx=None,
                           reuse=VARIABLE_SCOPE):
    """Register a call to tf.nn.convolution().

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [..filter_spatial_size..,
        in_channels, out_channels].  Bias should have shape [out_channels].
      inputs: Tensor of shape [batch_size, ..input_spatial_size.., in_channels].
        Inputs to layer.
      outputs: Tensor of shape [batch_size, ..output_spatial_size..,
        out_channels].  Output produced by layer.
      padding: string. see tf.nn.conv2d for valid values.
      strides: List of ints of length len(..input_spatial_size..). Strides for
        convolution kernel in spatial dimensions.
      dilation_rate: List of ints of length len(..input_spatial_size..).
        Dilations along spatial dimension.
      data_format: str or None. Format of data.
      approx: str or None. If not None must be one of "kron" or "diagonal".
        The Fisher approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    # TODO(b/74793309): Have this use _get_block_type like the other
    # registration functions?
    assert approx is None or approx == APPROX_KRONECKER_NAME

    block = self.register_block(
        params,
        fb.ConvKFCBasicFB(
            layer_collection=self,
            params=params,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            data_format=data_format),
        reuse=reuse)
    block.register_additional_tower(inputs, outputs)

    self._add_uses(params, 1)

  def register_depthwise_conv2d(self,
                                params,
                                inputs,
                                outputs,
                                strides,
                                padding,
                                rate=None,
                                data_format=None,
                                approx=None,
                                reuse=VARIABLE_SCOPE):
    """Register a call to tf.nn.depthwise_conv2d().

    Args:
      params: 4-D Tensor of shape [filter_height, filter_width,
        in_channels, channel_multiplier].  Convolutional filter.
      inputs: Tensor of shape [batch_size, input_height, input_width,
        in_channels].  Inputs to layer.
      outputs: Tensor of shape [batch_size, output_height, output_width,
        in_channels * channel_multiplier].  Output produced by depthwise conv2d.
      strides: List of ints of length 4. Strides along all dimensions.
      padding: string. see tf.nn.conv2d for valid values.
      rate: None or List of ints of length 2. Dilation rates in spatial
        dimensions.
      data_format: str or None. Format of data.
      approx: str or None. If not None must "diagonal".  The Fisher
        approximation to use. If None the default value is used. (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    # TODO(b/74793309): Have this use _get_block_type like the other
    # registration functions?
    assert approx is None or approx == APPROX_DIAGONAL_NAME
    assert data_format in [None, "NHWC"]

    block = self.register_block(
        params,
        fb.DepthwiseConvDiagonalFB(
            layer_collection=self,
            params=params,
            strides=strides,
            padding=padding,
            rate=rate,
            data_format=data_format),
        reuse=reuse)
    block.register_additional_tower(inputs, outputs)

    self._add_uses(params, 1)

  def register_separable_conv2d(self,
                                depthwise_params,
                                pointwise_params,
                                inputs,
                                depthwise_outputs,
                                pointwise_outputs,
                                strides,
                                padding,
                                rate=None,
                                data_format=None,
                                approx=None,
                                reuse=VARIABLE_SCOPE):
    """Register a call to tf.nn.separable_conv2d().

    Note: This requires access to intermediate outputs between depthwise and
    pointwise convolutions.

    Args:
      depthwise_params: 4-D Tensor of shape [filter_height, filter_width,
        in_channels, channel_multiplier].  Filter for depthwise conv2d.
      pointwise_params: 4-D Tensor of shape [1, 1, in_channels *
        channel_multiplier, out_channels].  Filter for pointwise conv2d.
      inputs: Tensor of shape [batch_size, input_height, input_width,
        in_channels].  Inputs to layer.
      depthwise_outputs: Tensor of shape [batch_size, output_height,
        output_width, in_channels * channel_multiplier].  Output produced by
        depthwise conv2d.
      pointwise_outputs: Tensor of shape [batch_size, output_height,
        output_width, out_channels].  Output produced by pointwise conv2d.
      strides: List of ints of length 4. Strides for depthwise conv2d kernel in
        all dimensions.
      padding: string. see tf.nn.conv2d for valid values.
      rate: None or List of ints of length 2. Dilation rate of depthwise conv2d
        kernel in spatial dimensions.
      data_format: str or None. Format of data.
      approx: str or None. If not None must be one of "kron" or "diagonal".
        The Fisher approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    self.register_depthwise_conv2d(
        params=depthwise_params,
        inputs=inputs,
        outputs=depthwise_outputs,
        strides=strides,
        padding=padding,
        rate=rate,
        data_format=data_format,
        approx=APPROX_DIAGONAL_NAME,
        reuse=reuse)

    self.register_conv2d(
        params=pointwise_params,
        inputs=depthwise_outputs,
        outputs=pointwise_outputs,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format=data_format,
        approx=approx,
        reuse=reuse)

  def register_generic(self,
                       params,
                       batch_size,
                       approx=None,
                       reuse=VARIABLE_SCOPE):
    """Registers a generic layer.

    Args:
      params: Tensor or tuple of Tensors corresponding to the parameters.
      batch_size: 0-D Tensor. Size of the minibatch (for this tower).
      approx: str or None. It not None, must be one of "full" or "diagonal".
        The Fisher approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str. If True, this adds `batch_size` to the total
        mini-batch size use when estimating the Fisher block for this layer
        (which must have already been registered). If "VARIABLE_SCOPE", use
        tf.get_variable_scope().reuse. (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    block_type, approx = self._get_block_type(
        params, approx, self.default_generic_approximation,
        _GENERIC_APPROX_TO_BLOCK_TYPES)

    block = self.register_block(params, block_type(self, params), reuse=reuse)
    block.register_additional_tower(batch_size)

    self._add_uses(params, float("inf"))

  def register_fully_connected_multi(self, params, inputs, outputs,
                                     num_uses=None, approx=None,
                                     reuse=VARIABLE_SCOPE):
    """Register fully connected layers with shared parameters.

    This can handle general fully-connected layers with shared parameters, but
    has specialized approximations to deal with the case where there is a
    meaningful linear order to the share instances (such as in an RNN).

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [input_size, output_size].
        Bias should have shape [output_size].
      inputs: A list of Tensors, each of shape [batch_size, input_size]. Inputs
        to layer. The list indexes each use in the graph (which might
        correspond to a "time-step" in an RNN). OR, can be single Tensor, of
        shape [num_uses * batch_size , input_size], which is a reshaped version
        of a Tensor of shape [num_uses, batch_size, input_size].
      outputs: A list of Tensors, the same length as `inputs`, each of shape
        [batch_size, output_size]. Outputs produced by layer. The list indexes
        each use in the graph (which might correspond to a "time-step" in an
        RNN). Needs to correspond with the order used in `inputs`.  OR, can be
        a single Tensor of shape [num_uses * batch_size, output_size], which is
        a reshaped version of a Tensor of shape [num_uses, batch_size,
        output_size].
      num_uses: int or None. The number uses/time-steps in the graph where the
        layer appears. Only needed if both inputs and outputs are given in the
        single Tensor format. (Default: None)
      approx: str or None. If not None, must be of "kron_indep", "kron_series_1"
        or "kron_series_2". The Fisher approximation to use. If None the default
        value is used. (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.  (Note that the
        word `use` here has a completely different meaning to "use in the graph"
        as it perturns to the `inputs`, `outputs`, and `num_uses` arguments.)
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
    """
    block_type, approx = self._get_block_type(
        params, approx, self.default_fully_connected_multi_approximation,
        _FULLY_CONNECTED_MULTI_APPROX_TO_BLOCK_TYPES)

    # TODO(b/70283649): something along the lines of find_canonical_output
    # should be added back in here (and for the other block types, arguably).

    has_bias = isinstance(params, (tuple, list))
    block = self.register_block(params, block_type(self, has_bias=has_bias,
                                                   num_uses=num_uses),
                                reuse=reuse)
    block.register_additional_tower(inputs, outputs)
    if isinstance(inputs, (tuple, list)):
      assert len(inputs) == len(outputs)
      self._add_uses(params, len(inputs))
    else:
      self._add_uses(params, 1)

  def register_conv2d_multi(self,
                            params,
                            strides,
                            padding,
                            inputs,
                            outputs,
                            num_uses=None,
                            data_format=None,
                            dilations=None,
                            approx=None,
                            reuse=VARIABLE_SCOPE):
    """Registers convolutional layers with shared parameters.

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [kernel_height,
        kernel_width, in_channels, out_channels].  Bias should have shape
        [out_channels].
      strides: 1-D Tensor of length 4. Strides for convolution kernel.
      padding: string. see tf.nn.conv2d for valid values.
      inputs: A list of Tensors, each of shape [batch_size, height, width,
        in_channels]. Inputs to layer. The list indexes each use in the graph
        (which might correspond to a "time-step" in an RNN). OR, can be single
        Tensor, of shape [num_uses * batch_size, height, width, in_channels],
        which is a reshaped version of a Tensor of shape [num_uses, batch_size,
        height, width, in_channels].
      outputs: A list of Tensors, each of shape [batch_size, height, width,
        out_channels]. Output produced by layer. The list indexes each use
        in the graph (which might correspond to a "time-step" in an RNN).
        Needs to correspond with the order used in `inputs`.  OR, can be a
        single Tensor, of shape [num_uses * batch_size, height, width,
        out_channels], which is a reshaped version of a Tensor of shape
        [num_uses, batch_size, height, width, out_channels].
      num_uses: int or None. The number uses/time-steps in the graph where the
        layer appears. Only needed if both inputs and outputs are given in the
        single Tensor format. (Default: None)
      data_format: str or None. Format of data.
      dilations: List of 4 ints. Dilations along each dimension.
      approx: str or None. If not None must by "kron_indep". The Fisher
        approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.  (Note that the
        word `use` here has a completely different meaning to "use in the graph"
        as it perturns to the `inputs`, `outputs`, and `num_uses` arguments.)
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    block_type, approx = self._get_block_type(
        params, approx, self.default_conv2d_multi_approximation,
        _CONV2D_MULTI_APPROX_TO_BLOCK_TYPES)

    block = self.register_block(
        params,
        block_type(
            layer_collection=self,
            params=params,
            padding=padding,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilations,
            extract_patches_fn="extract_image_patches",
            num_uses=num_uses),
        reuse=reuse)

    block.register_additional_tower(inputs, outputs)
    if isinstance(inputs, (tuple, list)):
      assert len(inputs) == len(outputs)
      self._add_uses(params, len(inputs))
    else:
      self._add_uses(params, 1)

  # TODO(b/74108452): change the loss registration functions names to refer
  # to "loss functions" instead of distributions.  Following naming convention
  # of the loss function classes themselves.

  def register_embedding_multi(self,
                               params,
                               inputs,
                               outputs,
                               num_uses=None,
                               approx=None,
                               reuse=VARIABLE_SCOPE):
    """Registers embedding layers with shared parameters.

    Args:
      params: Embedding matrix of shape [vocab_size, embedding_size].
      inputs: A list of Tensors, each of shape [batch_size, input_size] and
        dtype int32. Indices into embedding matrix. The list indexes each use
        in the graph (which might correspond to a "time-step" in an RNN).
        OR, can be single Tensor, of shape [num_uses*batch_size, input_size],
        which is a reshaped version of a Tensor of shape [num_uses, batch_size,
        input_size].
      outputs: A list of Tensors, each of shape [batch_size, embedding_size].
        Outputs produced by layer. The list indexes each use in the graph
        (which might correspond to a "time-step" in an RNN). Needs to
        correspond with the order used in `inputs`. OR, can be a
        single Tensor, of shape [num_uses * batch_size, embedding_size], which
        is a reshaped version of a Tensor of shape [num_uses, batch_size,
        embedding_size].
      num_uses: int or None. The number uses/time-steps in the graph where the
        layer appears. Only needed if both inputs and outputs are given in the
        single Tensor format. (Default: None)
      approx: str or None. If not None must by "kron_indep". The Fisher
        approximation to use. If None the default value is used.
        (Default: None)
      reuse: bool or str.  If True, this adds `inputs` and `outputs` as an
        additional mini-batch/tower of data to use when estimating the Fisher
        block for this layer (which must have already been registered). If
        "VARIABLE_SCOPE", use tf.get_variable_scope().reuse.  (Note that the
        word `use` here has a completely different meaning to "use in the graph"
        as it perturns to the `inputs`, `outputs`, and `num_uses` arguments.)
        (Default: "VARIABLE_SCOPE")

    Raises:
      ValueError: For improper value to `approx`.
      KeyError: If reuse == True but no FisherBlock found for `params`.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    block_type, approx = self._get_block_type(
        params, approx, self.default_embedding_multi_approximation,
        _EMBEDDING_MULTI_APPROX_TO_BLOCK_TYPES)

    if isinstance(params, (tuple, list)):
      raise ValueError("Bias not supported.")
    vocab_size = int(params.shape[0])

    block = self.register_block(
        params, block_type(self, vocab_size, num_uses=num_uses), reuse=reuse)
    block.register_additional_tower(inputs, outputs)

    if isinstance(inputs, (tuple, list)):
      self._add_uses(params, len(inputs))
    else:
      self._add_uses(params, 1)

  def register_categorical_predictive_distribution(self,
                                                   logits,
                                                   seed=None,
                                                   targets=None,
                                                   name=None,
                                                   reuse=VARIABLE_SCOPE):
    """Registers a categorical predictive distribution.

    Args:
      logits: The logits of the distribution (i.e. its parameters).
      seed: The seed for the RNG (for debugging) (Default: None)
      targets: (OPTIONAL) The targets for the loss function.  Only required if
        one wants to call total_loss() instead of total_sampled_loss().
        total_loss() is required, for example, to estimate the
        "empirical Fisher" (instead of the true Fisher).
        (Default: None)
      name: (OPTIONAL) str or None. Unique name for this loss function. If None,
        a new name is generated. (Default: None)
      reuse: bool or str.  If True, this adds `logits` as an additional
        mini-batch/tower of inputs to the loss-function/predictive distribution
        (which must have already been registered). If "VARIABLE_SCOPE", use
        tf.get_variable_scope().reuse. (Default: "VARIABLE_SCOPE")
    """
    loss = lf.CategoricalLogitsNegativeLogProbLoss(logits, targets=targets,
                                                   seed=seed)
    self.register_loss_function(loss, logits,
                                "categorical_predictive_distribution",
                                name=name, reuse=reuse)

  def register_normal_predictive_distribution(self,
                                              mean,
                                              var=0.5,
                                              seed=None,
                                              targets=None,
                                              name=None,
                                              reuse=VARIABLE_SCOPE):
    """Registers a normal predictive distribution.

    Args:
      mean: The mean vector defining the distribution.
      var: The variance (must be a scalar).  Note that the default value of
        0.5 corresponds to a standard squared error loss (target -
        prediction)**2. If your squared error loss is of the form
        0.5*(target - prediction)**2 you should use var=1.0. (Default: 0.5)
      seed: The seed for the RNG (for debugging) (Default: None)
      targets: (OPTIONAL) The targets for the loss function.  Only required if
        one wants to call total_loss() instead of total_sampled_loss().
        total_loss() is required, for example, to estimate the
        "empirical Fisher" (instead of the true Fisher).
        (Default: None)
      name: (OPTIONAL) str or None. Unique name for this loss function. If None,
        a new name is generated. (Default: None)
      reuse: bool or str.  If True, this adds `mean` and `var` as an additional
        mini-batch/tower of inputs to the loss-function/predictive distribution
        (which must have already been registered). If "VARIABLE_SCOPE", use
        tf.get_variable_scope().reuse. (Default: "VARIABLE_SCOPE")
    """
    loss = lf.NormalMeanNegativeLogProbLoss(mean, var, targets=targets,
                                            seed=seed)
    self.register_loss_function(loss, mean,
                                "normal_predictive_distribution",
                                name=name, reuse=reuse)

  def register_multi_bernoulli_predictive_distribution(self,
                                                       logits,
                                                       seed=None,
                                                       targets=None,
                                                       name=None,
                                                       reuse=VARIABLE_SCOPE):
    """Registers a multi-Bernoulli predictive distribution.

    Args:
      logits: The logits of the distribution (i.e. its parameters).
      seed: The seed for the RNG (for debugging) (Default: None)
      targets: (OPTIONAL) The targets for the loss function.  Only required if
        one wants to call total_loss() instead of total_sampled_loss().
        total_loss() is required, for example, to estimate the
        "empirical Fisher" (instead of the true Fisher).
        (Default: None)
      name: (OPTIONAL) str or None. Unique name for this loss function. If None,
        a new name is generated. (Default: None)
      reuse: bool or str.  If True, this adds `logits` as an additional
        mini-batch/tower of inputs to the loss-function/predictive distribution
        (which must have already been registered). If "VARIABLE_SCOPE", use
        tf.get_variable_scope().reuse. (Default: "VARIABLE_SCOPE")
    """
    loss = lf.MultiBernoulliNegativeLogProbLoss(logits, targets=targets,
                                                seed=seed)
    self.register_loss_function(loss, logits,
                                "multi_bernoulli_predictive_distribution",
                                name=name, reuse=reuse)

  def make_or_get_factor(self, cls, args):
    """Insert `cls(args)` into 'self.fisher_factors` if not already present.

    Wraps constructor in `tf.variable_scope()` to ensure variables constructed
    in `cls.__init__` are placed under this LayerCollection's scope.

    Args:
      cls: Class that implements FisherFactor.
      args: Tuple of arguments to pass into `cls's constructor. Must be
        hashable.

    Returns:
      Instance of `cls` found in self.fisher_factors.
    """
    try:
      hash(args)
    except TypeError:
      raise TypeError(
          ("Unable to use (cls, args) = ({}, {}) as a key in "
           "LayerCollection.fisher_factors. The pair cannot be hashed.").format(
               cls, args))

    key = cls, args
    if key not in self.fisher_factors:
      with variable_scope.variable_scope(self._var_scope):
        self.fisher_factors[key] = cls(*args)
    return self.fisher_factors[key]

  @contextmanager
  def as_default(self):
    """Sets this LayerCollection as the default."""
    set_default_layer_collection(self)
    yield
    set_default_layer_collection(None)
