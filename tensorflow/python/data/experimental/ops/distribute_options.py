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
"""Experimental API for controlling distribution in `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.AutoShardPolicy")
class AutoShardPolicy(enum.IntEnum):
  """Represents the type of auto-sharding to use.

  OFF: No sharding will be performed.

  AUTO: Attempts FILE-based sharding, falling back to DATA-based sharding.

  FILE: Shards by input files (i.e. each worker will get a set of files to
  process). When this option is selected, make sure that there is at least as
  many files as workers. If there are fewer input files than workers, a runtime
  error will be raised.

  DATA: Shards by elements produced by the dataset. Each worker will process the
  whole dataset and discard the portion that is not for itself. Note that for
  this mode to correctly partitions the dataset elements, the dataset needs to
  produce elements in a deterministic order.

  HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
  placeholder to replace with `shard(num_workers, worker_index)`.
  """
  OFF = -1
  AUTO = 0
  FILE = 1
  DATA = 2
  HINT = 3

  @classmethod
  def _to_proto(cls, obj):
    """Convert enum to proto."""
    if obj == cls.OFF:
      return dataset_options_pb2.AutoShardPolicy.OFF
    if obj == cls.FILE:
      return dataset_options_pb2.AutoShardPolicy.FILE
    if obj == cls.DATA:
      return dataset_options_pb2.AutoShardPolicy.DATA
    if obj == cls.AUTO:
      return dataset_options_pb2.AutoShardPolicy.AUTO
    if obj == cls.HINT:
      return dataset_options_pb2.AutoShardPolicy.HINT
    raise ValueError("%s._to_proto() is called with undefined enum %s." %
                     (cls.__name__, obj.name))

  @classmethod
  def _from_proto(cls, pb):
    """Convert proto to enum."""
    if pb == dataset_options_pb2.AutoShardPolicy.OFF:
      return cls.OFF
    if pb == dataset_options_pb2.AutoShardPolicy.FILE:
      return cls.FILE
    if pb == dataset_options_pb2.AutoShardPolicy.DATA:
      return cls.DATA
    if pb == dataset_options_pb2.AutoShardPolicy.AUTO:
      return cls.AUTO
    if pb == dataset_options_pb2.AutoShardPolicy.HINT:
      return cls.HINT
    raise ValueError("%s._from_proto() is called with undefined enum %s." %
                     (cls.__name__, pb))


@tf_export("data.experimental.ExternalStatePolicy")
class ExternalStatePolicy(enum.Enum):
  """Represents how to handle external state during serialization.

  See the `tf.data.Options.experimental_external_state_policy` documentation
  for more information.
  """
  WARN = 0
  IGNORE = 1
  FAIL = 2

  @classmethod
  def _to_proto(cls, obj):
    """Convert enum to proto."""
    if obj == cls.IGNORE:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_IGNORE
    if obj == cls.FAIL:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_FAIL
    if obj == cls.WARN:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_WARN
    raise ValueError("%s._to_proto() is called with undefined enum %s." %
                     (cls.__name__, obj.name))

  @classmethod
  def _from_proto(cls, pb):
    """Convert proto to enum."""
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_IGNORE:
      return cls.IGNORE
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_FAIL:
      return cls.FAIL
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_WARN:
      return cls.WARN
    raise ValueError("%s._from_proto() is called with undefined enum %s." %
                     (cls.__name__, pb))


@tf_export("data.experimental.DistributeOptions")
class DistributeOptions(options.OptionsBase):
  """Represents options for distributed data processing.

  You can set the distribution options of a dataset through the
  `experimental_distribute` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.DistributeOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
  dataset = dataset.with_options(options)
  ```
  """

  auto_shard_policy = options.create_option(
      name="auto_shard_policy",
      ty=AutoShardPolicy,
      docstring="The type of sharding to use. See "
      "`tf.data.experimental.AutoShardPolicy` for additional information.",
      default_factory=lambda: AutoShardPolicy.AUTO)

  num_devices = options.create_option(
      name="num_devices",
      ty=int,
      docstring=
      "The number of devices attached to this input pipeline. This will be "
      "automatically set by `MultiDeviceIterator`.")

  def _to_proto(self):
    pb = dataset_options_pb2.DistributeOptions()
    pb.auto_shard_policy = AutoShardPolicy._to_proto(self.auto_shard_policy)  # pylint: disable=protected-access
    if self.num_devices is not None:
      pb.num_devices = self.num_devices
    return pb

  def _from_proto(self, pb):
    self.auto_shard_policy = AutoShardPolicy._from_proto(pb.auto_shard_policy)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_num_devices") is not None:
      self.num_devices = pb.num_devices
