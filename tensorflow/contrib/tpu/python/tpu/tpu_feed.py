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
# ===================================================================

"""Helper library for handling infeed between hosts and TPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_sharding

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops


class InfeedQueue(object):
  """A helper object to build a device infeed queue.

  The InfeedQueue builds the host-side and device-side Ops to enqueue and
  dequeue elements, respectively, and ensures that their types and
  shapes match.
  """

  def __init__(self,
               number_of_tuple_elements=None,
               tuple_types=None,
               tuple_shapes=None,
               shard_dimensions=None,
               name=None):
    """Creates a new InfeedQueue with the given configuration.

    The configuration need not be fully specified at creation since it
    can be modified subsequently by methods that set the values
    explicitly or infer them from the shapes of inputs.

    Args:
      number_of_tuple_elements: the number of Tensors fed atomically through the
        queue, must be present unless it can be inferred from other arguments.
      tuple_types: if not None, a list of types of the elements of the queue.
      tuple_shapes: if not None, a list of shapes of the elements of the queue.
      shard_dimensions: if not None, a list of dimensions on which the
        elements of the queue should be sharded during automatic
        parallelization.
      name: the name of the queue.

    Raises:
      ValueError: if number_of_tuple_elements <= 0; or
        number_of_tuple_arguments, tuple_types, tuple_shapes, and
        shard_dimensions are all None; or the length of tuple_types,
        tuple_shapes, or shard_dimensions is not equal to
        number_of_tuple_elements; or any element of shard_dimensions
        can't be converted to a Dimension.
      TypeError: if any element of tuple_types or tuple_shapes can't
        be converted to a dtype or TensorShape, respectively.
    """
    self._frozen = False
    self._generated_enqueue_ops = False
    self._generated_dequeue_op = False
    self._name = "InfeedQueue" if name is None else name
    if number_of_tuple_elements is None:
      if tuple_types is not None:
        number_of_tuple_elements = len(tuple_types)
      elif tuple_shapes is not None:
        number_of_tuple_elements = len(tuple_shapes)
      elif shard_dimensions is not None:
        number_of_tuple_elements = len(shard_dimensions)
      else:
        raise ValueError(
            "number of tuple elements cannot be inferred from InfeedQueue "
            "constructor"
        )
    if number_of_tuple_elements <= 0:
      raise ValueError("number_of_tuple_elements %d must be > 0" %
                       number_of_tuple_elements)
    # Make an empty sharding policy for each tuple element.
    self._sharding_policies = [
        tpu_sharding.ShardingPolicy()
        for _ in xrange(number_of_tuple_elements)
    ]
    if tuple_types is not None:
      self.set_tuple_types(tuple_types)
    else:
      self._tuple_types = None
    if tuple_shapes is not None:
      self.set_tuple_shapes(tuple_shapes)
    else:
      self._tuple_shapes = None
    if shard_dimensions is not None:
      self.set_shard_dimensions(shard_dimensions)
    self._validate()

  def _validate(self):
    """Checks that the configuration is self-consistent.

    Raises:
      ValueError: if the shapes and sharding policies don't match.
    """
    if self.tuple_shapes is not None:
      for (policy, shape) in zip(self._sharding_policies, self._tuple_shapes):
        # Raise an error if the policy is incompatible with the shape.
        _ = policy.get_sharded_shape(shape)

  @property
  def number_of_tuple_elements(self):
    """Returns the number of InfeedQueue tuple elements."""
    return len(self._sharding_policies)

  @property
  def tuple_types(self):
    """Returns the types of the InfeedQueue tuple elements."""
    return self._tuple_types

  def set_tuple_types(self, tuple_types):
    """Sets the type of each element of the queue.

    tuple_types must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a dtype.

    Args:
      tuple_types: the types of each queue element.

    Raises:
      ValueError: if tuple_types is not of length
        self.number_of_tuple_elements.
      TypeError: if an element of tuple_types cannot be converted to a
        dtype.
    """
    if len(tuple_types) != self.number_of_tuple_elements:
      raise ValueError("tuple_types is %s, but must be a list of length %d" %
                       (str(tuple_types), self.number_of_tuple_elements))
    if self._frozen:
      for (frozen, updated) in zip(self._tuple_types, tuple_types):
        if frozen != updated:
          raise ValueError(
              "Trying to update InfeedQueue with frozen configuration with an "
              "incompatible type. Frozen types are %s, updated types are %s" % (
                  str(self._tuple_types), str(tuple_types)))
    else:
      try:
        self._tuple_types = [dtypes.as_dtype(t) for t in tuple_types]
      except (TypeError) as e:
        raise TypeError(
            "tuple_types is %s, but must be a list of elements each "
            "convertible to dtype: got error %s" % (str(tuple_types), str(e)))

  @property
  def tuple_shapes(self):
    """Returns the shapes of the InfeedQueue tuple elements."""
    return self._tuple_shapes

  def set_tuple_shapes(self, tuple_shapes):
    """Sets the shape of each element of the queue.

    tuple_shapes must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a TensorShape.

    Args:
      tuple_shapes: the shapes of each queue element.

    Raises:
      ValueError: if tuple_shapes is not of length
        self.number_of_tuple_elements.
      TypeError: if an element of tuple_shapes cannot be converted to
        a TensorShape.
    """
    if len(tuple_shapes) != self.number_of_tuple_elements:
      raise ValueError("tuple_shapes is %s, but must be a list of length %d" %
                       (str(tuple_shapes), self.number_of_tuple_elements))
    try:
      tuple_shapes = [tensor_shape.as_shape(shape) for shape in tuple_shapes]
    except (ValueError, TypeError) as e:
      raise TypeError(
          "tuple_shapes is %s, but must be a list of elements each "
          "convertible to TensorShape: got error %s" % (str(tuple_shapes),
                                                        str(e)))
    if self._frozen:
      for (frozen, updated) in zip(self._tuple_shapes, tuple_shapes):
        if frozen != updated:
          raise ValueError(
              "Trying to update InfeedQueue with frozen configuration with an "
              "incompatible shape. Frozen shapes are %s, updated shapes are %s"
              % (str(self._tuple_shapes), str(tuple_shapes)))
    else:
      self._tuple_shapes = tuple_shapes
    self._validate()

  @property
  def sharding_policies(self):
    """Returns the sharding policies of the InfeedQueue tuple elements."""
    return self._sharding_policies

  @property
  def shard_dimensions(self):
    """Gets the shard dimension of each tuple element.

    Returns:
      A list of length number_of_tuple_elements, where each list entry
      is the shard dimension of that tuple element or None if the
      shard dimension has not been set.
    """
    # The number of shards is always the same for all the policies.
    return [policy.shard_dimension for policy in self._sharding_policies]

  def set_shard_dimensions(self, shard_dimensions):
    """Sets the shard_dimension of each element of the queue.

    shard_dimensions must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a Dimension compatible with self.tuple_shapes.

    Args:
      shard_dimensions: the dimensions of each queue element.

    Raises:
      ValueError: if shard_dimensions is not of length
        self.number_of_tuple_elements; or an element of
        shard_dimensions cannot be converted to a Dimension; or an
        element of shard_dimensions is a Dimension that is out of
        range for the corresponding tuple element shape.
    """
    if len(shard_dimensions) != self.number_of_tuple_elements:
      raise ValueError("shard_dimensions is %s, but must be a list of length %d"
                       % (str(shard_dimensions),
                          self.number_of_tuple_elements))
    for (policy, dimension) in zip(self._sharding_policies, shard_dimensions):
      policy.set_shard_dimension(dimension)
    self._validate()

  @property
  def number_of_shards(self):
    """Gets the number of shards to use for the InfeedQueue.

    Returns:
      Number of shards or None if the number of shards has not been set.
    """
    # The number of shards is always the same for all the policies.
    return self._sharding_policies[0].number_of_shards

  def set_number_of_shards(self, number_of_shards):
    """Sets the number of shards to use for the InfeedQueue.

    Args:
      number_of_shards: number of ways to shard the InfeedQueue.

    Raises:
      ValueError: if number_of_shards is not > 0; or the policies have
        been frozen and number_of_shards was already set to something
        else.
    """
    for policy in self._sharding_policies:
      policy.set_number_of_shards(number_of_shards)
    self._validate()

  def set_configuration_from_input_tensors(self, input_tensors):
    """Sets the shapes and types of the queue tuple elements.

    input_tensors is a list of Tensors whose types and shapes are used
    to set the queue configuration.

    Args:
      input_tensors: list of Tensors of the same types and shapes as
        the desired queue Tuple.

    Raises:
      ValueError: if input_tensors is not a list of length
        self.number_of_tuple_elements
    """
    if len(input_tensors) != self.number_of_tuple_elements:
      raise ValueError(
          "input_tensors is %s, but should be a list of %d Tensors", (
              str(input_tensors), self.number_of_tuple_elements))
    self.set_tuple_shapes([t.shape for t in input_tensors])
    self.set_tuple_types([t.dtype for t in input_tensors])

  def set_configuration_from_sharded_input_tensors(self, input_tensors):
    """Sets the shapes and types of the queue tuple elements.

    input_tensors is a list of lists of Tensors whose types and shapes are used
    to set the queue configuration. The length of the outer list is the number
    of shards required, and each inner list is the tuple of Tensors to use to
    determine the types and shapes of the corresponding shard. This method
    depends on the shard dimension, and calling it freezes the shard policy.

    Args:
      input_tensors: list of lists of Tensors. The outer list length corresponds
        to the desired number of shards, and each inner list is the size
        and shape of the desired configuration of the corresponding shard.

    Raises:
      ValueError: if any inner list is not a list of length
        self.number_of_tuple_elements; or the inner lists do not combine to
        form a consistent unsharded shape.
      TypeError: if the types of the Tensors in the inner lists do not match.
    """
    if not self._frozen:
      # Unset the tuple shapes in case the configuration becomes
      # transiently inconsistent.
      self._tuple_shapes = None
    number_of_shards = len(input_tensors)
    self.set_number_of_shards(number_of_shards)
    for t in input_tensors:
      if len(t) != self.number_of_tuple_elements:
        raise ValueError(
            "input_tensors is %s but must be a list of lists, where each inner"
            " list has length number_of_tuple_elements=%d" % (
                str(input_tensors), self.number_of_tuple_elements))
    # Transpose the inputs to make a list of shard shapes for each tuple
    # element.
    sharded_shapes = [[t[i].shape for t in input_tensors]
                      for i in xrange(self.number_of_tuple_elements)]
    # For each tuple, get the unsharded shape using that tuple's policy.
    unsharded_shapes = [
        policy.get_unsharded_shape(s)
        for (policy, s) in zip(self._sharding_policies, sharded_shapes)
    ]
    self.set_tuple_shapes(unsharded_shapes)
    for i in xrange(1, self.number_of_shards):
      for (t1, t2) in zip(input_tensors[0], input_tensors[i]):
        if t1.dtype != t2.dtype:
          raise TypeError(
              "types of the tuple elements of input_tensors %s are not "
              "consistent" % str(input_tensors))
    self.set_tuple_types([t.dtype for t in input_tensors[0]])

  def freeze(self):
    """Freezes the InfeedQueue so it can no longer be modified.

    The configuration is implicitly frozen before any host-side or
    device-side Ops are generated. The configuration cannot be frozen
    until the types and shapes of the tuple elements have been set.

    Raises:
      ValueError: if the types or shapes of the tuple elements have not been
      set.
    """
    self._frozen = True
    if self._tuple_types is None:
      raise ValueError(
          "Can't freeze an InfeedQueue without setting all tuple types.")
    if self._tuple_shapes is None:
      raise ValueError(
          "Can't freeze an InfeedQueue without setting all tuple shapes.")
    for shape in self._tuple_shapes:
      if shape.dims is None:
        raise ValueError(
            "Can't freeze an InfeedQueue without setting all tuple shapes.")
    for policy in self._sharding_policies:
      policy.freeze()
    self._validate()

  def generate_dequeue_op(self):
    """Generates the device-side Op to dequeue a tuple from the queue.

    Implicitly freezes the queue configuration if it is not already
    frozen, which will raise errors if the shapes and types have not
    been fully specified.

    Returns:
      A list of Outputs corresponding to a shard of infeed dequeued
      into XLA, suitable for use within a replicated block.

    Raises:
      ValueError: if the types or shapes of the tuple elements have not been
      set; or if a dequeue op has already been generated.
    """
    self.freeze()
    if self._generated_dequeue_op:
      raise ValueError("Can't generate two dequeue Ops from the same queue")
    self._generated_dequeue_op = True
    full_name = "%s/dequeue" % self._name
    sharded_shapes = [
        policy.get_sharded_shape(shape)
        for (shape, policy) in zip(self._tuple_shapes, self._sharding_policies)
    ]
    return tpu_ops.infeed_dequeue_tuple(
        dtypes=self._tuple_types, shapes=sharded_shapes, name=full_name)

  def _generate_enqueue_op(self,
                           inputs,
                           name_prefix,
                           index,
                           device=None,
                           tpu_ordinal=-1):
    """Generate a host-side Op to enqueue a tuple to the queue.

    If device is None the inputs are all required to have the same
    device specification, and the enqueue Op is colocated with
    inputs[0]. Otherwise the enqueue Op is placed on 'device'.

    Args:
      inputs: a list of Tensors with the types and shapes of the tuple elements.
      name_prefix: the base name for the Op.
      index: the shard index, used to uniquify the Op name.
      device: device to place the Op on, or None if it should be
        colocated with the inputs.
      tpu_ordinal: ordinal of the TPU device on the host to use for
      infeed if device is a CPU device. Should be set to -1 if device
      is a TPU device.

    Returns:
      An Op corresponding to a shard of infeed enqueued at the host,
      suitable for use within a replicated block.

    Raises:
      ValueError: if device is None and inputs do not all have the
        same device specification.
    """
    full_name = "%s/%d" % (name_prefix, index)
    shapes = [t.shape for t in inputs]
    if device is None:
      devices = [t.device for t in inputs]
      for i in xrange(1, self.number_of_tuple_elements):
        if devices[0] != devices[i]:
          raise ValueError(
              "input devices for shard %d are %s, but should all be the same",
              index, str(devices))
      with ops.colocate_with(inputs[0]):
        return tpu_ops.infeed_enqueue_tuple(
            inputs=inputs,
            shapes=shapes,
            name=full_name,
            device_ordinal=tpu_ordinal)
    else:
      with ops.device(device):
        return tpu_ops.infeed_enqueue_tuple(
            inputs=inputs,
            shapes=shapes,
            name=full_name,
            device_ordinal=tpu_ordinal)

  def generate_enqueue_ops(self, sharded_inputs, tpu_ordinal_function=None):
    """Generates the host-side Ops to enqueue the shards of a tuple.

    sharded_inputs is a list, one for each shard, of lists of
    Tensors. sharded_inputs[0] is the tuple of Tensors to use to feed
    shard 0 if the queue. Returns the host-side Ops that must be run to
    enqueue the sharded tuple. The Op for shard i is colocated with the inputs
    for shard i.

    Implicitly freezes the queue configuration if it is not already
    frozen. If the configuration has already been frozen, and is not
    compatible with the types and shapes of sharded_inputs, an error
    will be raised.

    Args:
      sharded_inputs: a list of lists of Tensors. The length of the outer list
        determines the number of shards. Each inner list indicates the types
        and shapes of the tuples in the corresponding shard.
      tpu_ordinal_function: if not None, a function that takes the
        shard index as input and returns the ordinal of the TPU device
        the shard's infeed should be placed on. tpu_ordinal_function must be
        set if the inputs are placed on CPU devices.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the shapes of the elements of sharded_inputs
        don't form a consistent unsharded tuple; or if the elements of a tuple
        have different device constraints.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the types of the elements of sharded_inputs
        don't form a consistent unsharded tuple.
    """
    self.set_configuration_from_sharded_input_tensors(sharded_inputs)
    self.freeze()
    if self._generated_enqueue_ops:
      raise ValueError("Can't generate two enqueue Ops from the same queue")
    self._generated_enqueue_ops = True
    if tpu_ordinal_function is None:
      tpu_ordinal_function = lambda index: -1
    name_prefix = "%s/enqueue" % self._name
    return [
        self._generate_enqueue_op(shard, name_prefix, index,
                                  tpu_ordinal=tpu_ordinal_function(index))
        for (shard, index) in zip(sharded_inputs, xrange(self.number_of_shards))
    ]

  # TODO(misard) Generalize this to the case of systems that don't
  # have 8 devices per host, and figure out what to do with
  # model-parallelism.
  def _default_placement_function(self, index):
    return "/task:%d/device:CPU:0" % (index / 8)

  def _default_ordinal_function(self, index):
    return index % 8

  # TODO(b/36470756) remove this from tutorials once we have a better story
  # for automatic placement of input pipelines.
  def split_inputs_and_generate_enqueue_ops(self,
                                            inputs,
                                            global_tpu_id=None,
                                            placement_function=None,
                                            tpu_ordinal_function=None):
    """POORLY-PERFORMING ON MULTI-HOST SYSTEMS.

    Generates the host-side Ops to enqueue a tuple.

    This method performs poorly because it takes an entire input on a single
    host, splits it, and distributes it to all of the cores. It is present only
    to simplify tutorial examples.

    inputs is a list of Tensors to use to feed the queue. Each input is split
    into self.number_of_shards shards. Returns an Op for each shard to enqueue
    the shard. The Op for shard i is placed on device placement_function(i).

    Implicitly freezes the queue configuration if it is not already
    frozen. If the configuration has already been frozen, and is not
    compatible with the types and shapes of inputs, an error
    will be raised.

    Args:
      inputs: a list of Tensors which indicates the types and shapes of the
        queue tuple.
     global_tpu_id: if not None, a Numpy 2D array indicating the global
        id of each TPU device in the system. The outer dimension of the
        array is host task id, and the inner dimension is device ordinal,
        so e.g., global_tpu_id[x][y] indicates the global id of device
        /task:x/device:TPU_NODE:y. If global_tpu_id is not None, but
        placement_function and ordinal_function are None, then global_tpu_id
        will be used to place infeed on the TPUs with the first k global ids,
        where k is the number of shards in the queue.
      placement_function: if not None, a function that takes the shard
        index as input and returns a device string indicating which
        device the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.
      tpu_ordinal_function: if not None, a function that takes the
        shard index as input and returns the ordinal of the TPU device
        the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of inputs are not compatible with the frozen
        configuration.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of inputs are not compatible with the frozen
        configuration.
    """
    if global_tpu_id is None:
      if placement_function is None:
        placement_function = self._default_placement_function
      if tpu_ordinal_function is None:
        tpu_ordinal_function = self._default_ordinal_function
    else:
      global_id_map = {}
      for host, devices in enumerate(global_tpu_id):
        for ordinal, global_id in enumerate(devices):
          global_id_map[global_id] = (host, ordinal)

      def _placement_function_from_map(index):
        return "/task:%d/device:CPU:0" % global_id_map[index][0]

      def _ordinal_function_from_map(index):
        return global_id_map[index][1]

      if placement_function is None:
        placement_function = _placement_function_from_map
      if tpu_ordinal_function is None:
        tpu_ordinal_function = _ordinal_function_from_map
    self.set_configuration_from_input_tensors(inputs)
    self.freeze()
    if self._generated_enqueue_ops:
      raise ValueError("Can't generate two enqueue Ops from the same queue")
    self._generated_enqueue_ops = True
    split_name_prefix = "%s/split" % self._name
    if self.number_of_shards == 1:
      transposed_sharded_inputs = [[inp] for inp in inputs]
    else:
      transposed_sharded_inputs = [
          array_ops.split(
              inp,
              self.number_of_shards,
              axis=policy.shard_dimension,
              name="%s/%d" % (split_name_prefix, index))
          for (inp, policy, index) in zip(inputs, self._sharding_policies,
                                          xrange(self.number_of_tuple_elements))
      ]
    sharded_inputs = [[shard[i] for shard in transposed_sharded_inputs]
                      for i in xrange(self.number_of_shards)]
    name_prefix = "%s/enqueue" % self._name
    return [
        self._generate_enqueue_op(
            shard,
            name_prefix,
            index,
            device=placement_function(index),
            tpu_ordinal=tpu_ordinal_function(index))
        for (shard, index) in zip(sharded_inputs, xrange(self.number_of_shards))
    ]
