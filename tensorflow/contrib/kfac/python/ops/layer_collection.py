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

# Possible value for 'reuse' keyword argument. Sets 'reuse' to
# tf.get_variable_scope().reuse.
VARIABLE_SCOPE = "VARIABLE_SCOPE"

# TODO(jamesmartens): need to add find_canonical_output back into this somewhere


def ensure_sequence(obj):
  """If `obj` isn't a tuple or list, return a tuple containing `obj`."""
  if isinstance(obj, (tuple, list)):
    return obj
  else:
    return (obj,)


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
  """

  def __init__(self,
               graph=None,
               colocate_cov_ops_with_inputs=False,
               name="LayerCollection"):
    self.fisher_blocks = LayerParametersDict()
    self.fisher_factors = OrderedDict()
    self._linked_parameters = dict(
    )  # dict mapping sets of variables to optionally specified approximations.
    self._graph = graph or ops.get_default_graph()
    self._loss_dict = {}  # {str: LossFunction}
    self._subgraph = None
    self._default_generic_approximation = APPROX_FULL_NAME
    self._default_fully_connected_approximation = APPROX_KRONECKER_NAME
    self._default_convolution_2d_approximation = APPROX_KRONECKER_NAME
    self._colocate_cov_ops_with_inputs = colocate_cov_ops_with_inputs

    with variable_scope.variable_scope(None, default_name=name) as scope:
      self._var_scope = scope.name

  @property
  def losses(self):
    """LossFunctions registered with this LayerCollection."""
    return list(self._loss_dict.values())

  def is_variable_registered(self, variable):
    """Checks whether the variable has already been registered.

    Args:
      variable: A single variable or tensor.
    Returns:
      True if the variable has been registered either by itself or as part of a
      tuple.
    """
    return any([
        variable in key if isinstance(key, (tuple, list)) else variable == key
        for key in self.fisher_blocks.keys()
    ])

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
  def default_generic_approximation(self):
    return self._default_generic_approximation

  @default_generic_approximation.setter
  def default_generic_approximation(self, value):
    if value not in _GENERIC_APPROX_TO_BLOCK_TYPES:
      raise ValueError(
          "{} is not a valid approximation for generic variables.".format(
              value))
    self._default_generic_approximation = value

  @property
  def default_fully_connected_approximation(self):
    return self._default_fully_connected_approximation

  @default_fully_connected_approximation.setter
  def default_fully_connected_approximation(self, value):
    if value not in _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES:
      raise ValueError(
          "{} is not a valid approximation for fully connected layers.".format(
              value))
    self._default_fully_connected_approximation = value

  @property
  def default_conv2d_approximation(self):
    return self._default_convolution_2d_approximation

  @default_conv2d_approximation.setter
  def default_conv2d_approximation(self, value):
    if value not in _CONV2D_APPROX_TO_BLOCK_TYPES:
      raise ValueError(
          "{} is not a valid approximation for 2d convolutional layers.".format(
              value))
    self._default_convolution_2d_approximation = value

  def register_block(self, layer_key, fisher_block, reuse=VARIABLE_SCOPE):
    """Validates and registers the layer_key associated with the fisher_block.

    Args:
      layer_key: A variable or tuple of variables. The key to check for in
          existing registrations and to register if valid.
      fisher_block: The associated `FisherBlock`.
      reuse: Method to use for inserting new `FisherBlock`s. One of True, False,
        or VARIABLE_SCOPE.

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
        for var in ensure_sequence(params)
    }
    for variable in ensure_sequence(layer_key):
      if variable in variable_to_block:
        prev_key, prev_block = variable_to_block[variable]
        raise ValueError(
            "Attempted to register layer_key {} with block {}, but variable {}"
            " was already registered in key {} with block {}.".format(
                layer_key, fisher_block, variable, prev_key, prev_block))
    self.fisher_blocks[layer_key] = fisher_block
    return fisher_block

  def get_use_count_map(self):
    """Returns a dict of variables to their number of registrations."""
    vars_to_uses = defaultdict(int)
    for key, block in six.iteritems(self.fisher_blocks):
      key = key if isinstance(key, (tuple, list)) else (key,)
      for k in key:
        vars_to_uses[k] += block.num_registered_minibatches
    return vars_to_uses

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
    params = frozenset(ensure_sequence(params))

    # Check if any of the variables in 'params' is already in
    # 'self.fisher_blocks.keys()'.
    for registered_params, fisher_block in self.fisher_blocks.items():
      registered_params_set = set(ensure_sequence(registered_params))
      for variable in params:
        if (variable in registered_params_set and
            params != registered_params_set):
          raise ValueError(
              "Can't link parameters {}, variable {} was already registered in "
              "group {} with layer {}".format(params, variable,
                                              registered_params, fisher_block))

    # Check if any of the variables in 'params' is already in
    # 'self.linked_parameters'.
    for variable in params:
      for other_linked_params in self.linked_parameters:
        if variable in other_linked_params:
          raise ValueError("Can't link parameters {}, variable {} was already "
                           "linked in group {}.".format(params, variable,
                                                        other_linked_params))
    self._linked_parameters[params] = approximation

  def create_subgraph(self):
    if not self.losses:
      raise ValueError("Must have at least one registered loss.")
    inputs_to_losses = nest.flatten(tuple(loss.inputs for loss in self.losses))
    self._subgraph = utils.SubGraph(inputs_to_losses)

  def total_loss(self):
    return math_ops.add_n(tuple(loss.evaluate() for loss in self.losses))

  def total_sampled_loss(self):
    return math_ops.add_n(
        tuple(loss.evaluate_on_sample() for loss in self.losses))

  def _get_linked_approx(self, params):
    """If params were linked, return their specified approximation."""
    params_set = frozenset(ensure_sequence(params))
    if params_set in self.linked_parameters:
      return self.linked_parameters[params_set]
    else:
      return None

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
      outputs: Tensor of shape [batch_size, output_size]. Preactivations
        produced by layer.
      approx: str. One of APPROX_KRONECKER_NAME or APPROX_DIAGONAL_NAME.
      reuse: bool or str.  If True, reuse an existing FisherBlock. If False,
        create a new FisherBlock.  If VARIABLE_SCOPE, use
        tf.get_variable_scope().reuse.

    Raises:
      ValueError: For improper value to 'approx'.
      KeyError: If reuse == True but no FisherBlock found for 'params'.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """
    if approx is None:
      approx = self._get_linked_approx(params)
      if approx is None:
        approx = self.default_fully_connected_approximation

    if approx not in _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES:
      raise ValueError("Bad value {} for approx.".format(approx))

    block_type = _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES[approx]
    has_bias = isinstance(params, (tuple, list))

    block = self.register_block(params, block_type(self, has_bias), reuse=reuse)
    block.register_additional_minibatch(inputs, outputs)

  def register_conv2d(self,
                      params,
                      strides,
                      padding,
                      inputs,
                      outputs,
                      approx=None,
                      reuse=VARIABLE_SCOPE):
    """Registers a convolutional layer.

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [kernel_height,
        kernel_width, in_channels, out_channels].  Bias should have shape
        [out_channels].
      strides: 1-D Tensor of length 4. Strides for convolution kernel.
      padding: string. see tf.nn.conv2d for valid values.
      inputs: Tensor of shape [batch_size, height, width, in_channels]. Inputs
        to layer.
      outputs: Tensor of shape [batch_size, height, width, out_channels].
        Preactivations produced by layer.
      approx: str. One of APPROX_KRONECKER_NAME or APPROX_DIAGONAL_NAME.
      reuse: bool or str.  If True, reuse an existing FisherBlock. If False,
        create a new FisherBlock.  If VARIABLE_SCOPE, use
        tf.get_variable_scope().reuse.

    Raises:
      ValueError: For improper value to 'approx'.
      KeyError: If reuse == True but no FisherBlock found for 'params'.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """

    if approx is None:
      approx = self._get_linked_approx(params)
      if approx is None:
        approx = self.default_conv2d_approximation

    if approx not in _CONV2D_APPROX_TO_BLOCK_TYPES:
      raise ValueError("Bad value {} for approx.".format(approx))

    block_type = _CONV2D_APPROX_TO_BLOCK_TYPES[approx]
    block = self.register_block(
        params, block_type(self, params, strides, padding), reuse=reuse)
    block.register_additional_minibatch(inputs, outputs)

  def register_generic(self,
                       params,
                       batch_size,
                       approx=None,
                       reuse=VARIABLE_SCOPE):
    """Registers a generic layer.

    Args:
      params: Tensor or 2-tuple of Tensors corresponding to weight and bias of
        this layer. Weight matrix should have shape [kernel_height,
        kernel_width, in_channels, out_channels].  Bias should have shape
        [out_channels].
      batch_size: 0-D Tensor. Size of the minibatch.
      approx: str. One of APPROX_KRONECKER_NAME or APPROX_DIAGONAL_NAME.
      reuse: bool or str.  If True, reuse an existing FisherBlock. If False,
        create a new FisherBlock.  If VARIABLE_SCOPE, use
        tf.get_variable_scope().reuse.

    Raises:
      ValueError: For improper value to 'approx'.
      KeyError: If reuse == True but no FisherBlock found for 'params'.
      ValueError: If reuse == True and FisherBlock found but of the wrong type.
    """

    if approx is None:
      approx = self._get_linked_approx(params)
      if approx is None:
        approx = self.default_generic_approximation

    if approx not in _GENERIC_APPROX_TO_BLOCK_TYPES:
      raise ValueError("Bad value {} for approx.".format(approx))

    block_type = _GENERIC_APPROX_TO_BLOCK_TYPES[approx]
    block = self.register_block(params, block_type(self, params), reuse=reuse)
    block.register_additional_minibatch(batch_size)

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
      reuse: (OPTIONAL) bool or str.  If True, reuse an existing FisherBlock.
        If False, create a new FisherBlock.  If VARIABLE_SCOPE, use
        tf.get_variable_scope().reuse.

    Raises:
      ValueError: If reuse == True and name == None.
      ValueError: If reuse == True and seed != None.
      KeyError: If reuse == True and no existing LossFunction with 'name' found.
      KeyError: If reuse == False and existing LossFunction with 'name' found.
    """
    name = name or self._graph.unique_name(
        "register_categorical_predictive_distribution")

    if reuse == VARIABLE_SCOPE:
      reuse = variable_scope.get_variable_scope().reuse

    if reuse:
      if name is None:
        raise ValueError(
            "If reuse is enabled, loss function's name must be set.")
      if seed is not None:
        raise ValueError(
            "Seed can only be specified at LossFunction instantiation.")

      loss = self._loss_dict.get(name, None)

      if loss is None:
        raise KeyError(
            "Unable to find loss function named {}. Create a new LossFunction "
            "with reuse=False.".format(name))

      loss.register_additional_minibatch(logits, targets=targets)
    else:
      if name in self._loss_dict:
        raise KeyError(
            "Loss function named {} already exists. Set reuse=True to append "
            "another minibatch.".format(name))
      loss = lf.CategoricalLogitsNegativeLogProbLoss(
          logits, targets=targets, seed=seed)
      self._loss_dict[name] = loss

  def register_normal_predictive_distribution(self,
                                              mean,
                                              var=0.5,
                                              seed=None,
                                              targets=None,
                                              name=None):
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
    """
    name = name or self._graph.unique_name(
        "register_normal_predictive_distribution")
    if name in self._loss_dict:
      raise NotImplementedError(
          "Adding logits to an existing LossFunction not yet supported.")
    loss = lf.NormalMeanNegativeLogProbLoss(
        mean, var, targets=targets, seed=seed)
    self._loss_dict[name] = loss

  def register_multi_bernoulli_predictive_distribution(self,
                                                       logits,
                                                       seed=None,
                                                       targets=None,
                                                       name=None):
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
    """
    name = name or self._graph.unique_name(
        "register_multi_bernoulli_predictive_distribution")
    if name in self._loss_dict:
      raise NotImplementedError(
          "Adding logits to an existing LossFunction not yet supported.")
    loss = lf.MultiBernoulliNegativeLogProbLoss(
        logits, targets=targets, seed=seed)
    self._loss_dict[name] = loss

  def make_or_get_factor(self, cls, args):
    """Insert 'cls(args)' into 'self.fisher_factors' if not already present.

    Wraps constructor in 'tf.variable_scope()' to ensure variables constructed
    in 'cls.__init__' are placed under this LayerCollection's scope.

    Args:
      cls: Class that implements FisherFactor.
      args: Tuple of arguments to pass into 'cls's constructor. Must be
        hashable.

    Returns:
      Instance of 'cls' found in self.fisher_factors.
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
      colo = self._colocate_cov_ops_with_inputs
      with variable_scope.variable_scope(self._var_scope):
        self.fisher_factors[key] = cls(*args, colocate_cov_ops_with_inputs=colo)
    return self.fisher_factors[key]
