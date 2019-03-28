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
# =============================================================================
"""Contains class for creating input infeeds into TF graphs targeting the IPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import ops


class IPUInfeedQueue:
  """Wraps a tf.Dataset object with infeed operations specific to the IPU.

  This class, along with `contrib.ipu.loops` is used to create a data pipeline
  from a `dataset` into a training/inference loop on the IPU inside a single
  `session.run` which reduces the overheads of calling `session.run` for each
  iteration of the loop.

  You should pass the infeed queue as an argument to a loop from
  `contrib.ipu.loops`. These loops will then handle the dequeuing of the data to
  the device automatically.

  The following skeleton shows how to use this method when building a training
  loop - note how the body signature contains variables which correspond to the
  nested structure of `tf.Tensor`s representing the next element in the infeed
  queue:

  ```python
  # Create an example dataset.
  dataset = ...  # A `tf.data.Dataset` object.

  def dataset_parser(value):
    features, labels = parse_record(value)
    return {"features": features,
            "labels": labels}
  # The resulting dataset has a nested structure of: {features, labels}.
  dataset = dataset.map(dataset_parser)

  infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

  # dataset can no longer be used beyond this point.

  def my_net():
    # Note how the nested structure forms part of the loop body signature.
    def body(loss, features, labels):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = tf.conv2d(features, .....)
        ...
        ...
        logits = tf.nn.xw_plus_b(....)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
      optimizer = gradient_descent.GradientDescentOptimizer(0.000001)
      train = optimizer.minimize(loss)
      with ops.control_dependencies([train]):
        return array_ops.identity(loss)

    loss = 0.0
    return = tf.conrib.ipu.loops.repeat(10000, body, [loss], infeed_queue)

  with ipu.ops.ipu_scope("/device:IPU:0"):
    res = ipu_compiler.compile(my_net, inputs=[])

  with tf.Session() as sess:
    sess.run(infeed_queue.initializer)
    sess.run(variables.global_variables_initializer())
    result = sess.run(res)
  ```
  """

  def __init__(self, dataset, device_ordinal=0, replication_factor=1):
    """Creates an IPUInfeedQueue object.

    Args:
       dataset: a tf.data.Dataset object, all transformations e.g. `shuffle`,
        `repeat`, `batch` must be applied prior to passing in to this function.
        This dataset can no longer be used after creating this queue.
       device_ordinal: ordinal of the device on which this queue will be used.
       replication_factor: the number of replicated graphs this infeed will be
         used in.

    Raises:
      ValueError: if all dimensions of shapes of dataset.output_shapes are not
        fully defined. tf.data.batch function must be called with
        `drop_remainder=True` to ensure that batch size is constant.

    """

    for output_shape in dataset_ops.flat_structure(dataset)["output_shapes"]:
      if isinstance(output_shape, list) or isinstance(output_shape, tuple):
        raise ValueError("Nested list/tuple input shapes are not supported")
      if not output_shape.is_fully_defined():
        raise ValueError("""Output shape {} is not fully defined. If using \
tf.Dataset.batch, set `drop_remainder=True`.""".format(output_shape))

    with ops.device('/device:CPU:0'):
      # Apply the dataset and take ownership.
      dataset = dataset._apply_options()
      self._structure = dataset._element_structure
      self._flat_structure = dataset_ops.flat_structure(dataset)
      # Batch the dataset to take replication into account.
      self._dataset = dataset.batch(replication_factor, drop_remainder=True)
      try:
        ds_variant = self._dataset._variant_tensor
      except TypeError:
        ds_variant = self._dataset._as_variant_tensor
      # ID used for differentiating between datasets.
      self._id = str(id(ds_variant))

      # Dataset iterator creator.
      self._initializer = gen_pop_datastream_ops.ipu_consume_dataset(
          input_dataset=ds_variant,
          id=self._id,
          device_ordinal=device_ordinal,
          **dataset_ops.flat_structure(self._dataset))

    self._dequeued = False

  def _dequeue(self):
    """Returns a nested structure of `tf.Tensor`s representing the next element
    in the infeed queue.

    This function should not be called directly, instead the infeed should be
    passed to a loop from `contrib.ipu.loops`.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    flat_ret = gen_pop_datastream_ops.pop_datastream_infeed_dequeue(
        infeed_id=self._id, **self._flat_structure)
    self._dequeued = True
    return self._structure._from_tensor_list(flat_ret)

  @property
  def dequeued(self):
    """ Returns whether this queue has been dequeued.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    return self._dequeued

  @property
  def number_of_tuple_elements(self):
    """Returns the number of IPUInfeedQueue tuple elements."""
    return len(self._structure._flat_shapes)

  @property
  def initializer(self):
    """A `tf.Operation` that should be run to initialize this IPUInfeedQueue.

    Returns:
      A `tf.Operation` that should be run to initialize this IPUInfeedQueue
    """
    return self._initializer

  def get_next(self):
    """Obsolete function."""
    raise ValueError("""`get_next()` is now obsolete as the IPUInfeedQueue is \
now automatically dequeued by the loop.""")
