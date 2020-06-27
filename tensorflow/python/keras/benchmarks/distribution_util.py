# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions for running model in distributed settings.
   Reference: https://github.com/tensorflow/models/blob/master/official/utils/misc/distribution_utils.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import distribute
from tensorflow.python.distribute import cross_device_ops
from tensorflow.python.distribute.collective_all_reduce_strategy import \
  CollectiveAllReduceStrategy


def _collective_communication(all_reduce_alg):
  """Return a CollectiveCommunication based on all_reduce_alg.

  Args:
    all_reduce_alg: A string specifying which collective communication to pick,
      or None.

  Returns:
    `distribute.experimental.CollectiveCommunication` object.

  Raises:
    ValueError: If `all_reduce_alg` not in [None, "ring", "nccl"].
  """
  collective_communication_options = {
      None: cross_device_ops.CollectiveCommunication.AUTO,
      "ring": cross_device_ops.CollectiveCommunication.RING,
      "nccl": cross_device_ops.CollectiveCommunication.NCCL
  }
  if all_reduce_alg not in collective_communication_options:
    raise ValueError(
        "When used with `multi_worker_mirrored`, valid values for "
        "all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}".format(
            all_reduce_alg))
  return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
  """Return a CrossDeviceOps based on all_reduce_alg and num_packs.

  Args:
    all_reduce_alg: A string specifying which cross device op to pick, or None.
    num_packs: an integer specifying number of packs for the cross device op.

  Returns:
    `distribute.CrossDeviceOps` object or None.

  Raises:
    ValueError: If `all_reduce_alg` not in [None, "nccl", "hierarchical_copy"].
  """
  if all_reduce_alg is None:
    return None
  mirrored_all_reduce_options = {
      "nccl": cross_device_ops.NcclAllReduce,
      "hierarchical_copy": cross_device_ops.HierarchicalCopyAllReduce
  }
  if all_reduce_alg not in mirrored_all_reduce_options:
    raise ValueError(
        "When used with `mirrored`, valid values for all_reduce_alg are "
        "[`nccl`, `hierarchical_copy`].  Supplied value: {}".format(
            all_reduce_alg))
  cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
  return cross_device_ops_class(num_packs=num_packs)


def get_distribution_strategy(distribution_strategy="mirrored",
                              num_gpus=0,
                              all_reduce_alg=None,
                              num_packs=1,
                              tpu_address=None):
  """Return a DistributionStrategy for running the model.

  Args:
    distribution_strategy: A string specifying which distribution strategy to
      use. Accepted values are "off", "one_device", "mirrored",
      "parameter_server", "multi_worker_mirrored", and "tpu" --
      case insensitive.
      "off" means not to use Distribution Strategy; "tpu" means to use
      TPUStrategy using `tpu_address`.
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Optional. Specifies which algorithm to use when performing
      all-reduce. For `MirroredStrategy`, valid values are "nccl" and
      "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
      "ring" and "nccl".  If None, DistributionStrategy will choose based on
      device topology.
    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
    tpu_address: Optional. String that represents TPU to connect to. Must not
      be None if `distribution_strategy` is set to `tpu`.
  Returns:
    `distribute.DistibutionStrategy` object.
  Raises:
    ValueError: If `distribution_strategy` is "off" or "one_device" and
      `num_gpus` is larger than 1; or `num_gpus` is negative or if
      `distribution_strategy` is `tpu` but `tpu_address` is not specified.
  """
  if (num_gpus < 0):
    raise ValueError("`num_gpus` can nnot be negative.")

  distribution_strategy = distribution_strategy.lower()
  if distribution_strategy == "off":
    if num_gpus > 1:
      raise ValueError(
          "When {} GPUs are specified, distribution_strategy "
          "flag cannot be set to `off`.".format(num_gpus))
    return None
  # TODO: `TPU`, `parameter_server`.

  if distribution_strategy == 'multi_worker_mirrored':
    return CollectiveAllReduceStrategy(
        communication=_collective_communication(all_reduce_alg))

  if distribution_strategy == "one_device":
    if num_gpus == 0:
      return distribute.one_device_strategy.OneDeviceStrategy("device:CPU:0")
    if num_gpus > 1:
      raise ValueError("`OneDeviceStrategy` can not be used for more than "
                       "one device.")
    return distribute.one_device_strategy.OneDeviceStrategy("device:GPU:0")

  if distribution_strategy == "mirrored":
    if num_gpus == 0:
      devices = ["device:CPU:0"]
    else:
      devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    return distribute.mirrored_strategy.MirroredStrategy(
        devices=devices,
        cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

  raise ValueError(
      "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def get_strategy_scope(strategy):
  if strategy:
    strategy_scope = strategy.scope()
  else:
    strategy_scope = DummyContextManager()

  return strategy_scope


class DummyContextManager(object):
  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass
