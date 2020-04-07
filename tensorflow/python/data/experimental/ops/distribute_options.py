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

from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.AutoShardPolicy")
class AutoShardPolicy(enum.IntEnum):
  """Represents the type of auto-sharding we enable.

  Please see the DistributeOptions.auto_shard_policy documentation for more
  information on each type of autosharding.
  """
  OFF = -1
  AUTO = 0
  FILE = 1
  DATA = 2


class ExternalStatePolicy(enum.Enum):
  WARN = 0
  IGNORE = 1
  FAIL = 2


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
      docstring="The type of sharding that auto-shard should attempt. If this "
      "is set to FILE, then we will attempt to shard by files (each worker "
      "will get a set of files to process). If we cannot find a set of files "
      "to shard for at least one file per worker, we will error out. When this "
      "option is selected, make sure that you have enough files so that each "
      "worker gets at least one file. There will be a runtime error thrown if "
      "there are insufficient files. "
      "If this is set to DATA, then we will shard by elements produced by the "
      "dataset, and each worker will process the whole dataset and discard the "
      "portion that is not for itself. "
      "If this is set to OFF, then we will not autoshard, and each worker will "
      "receive a copy of the full dataset. "
      "This option is set to AUTO by default, AUTO will attempt to first shard "
      "by FILE, and fall back to sharding by DATA if we cannot find a set of "
      "files to shard.",
      default_factory=lambda: AutoShardPolicy.AUTO)

  _make_stateless = options.create_option(
      name="_make_stateless",
      ty=bool,
      docstring=
      "Determines whether the input pipeline should be rewritten to not "
      "contain stateful transformations (so that its graph can be moved "
      "between devices).")

  num_devices = options.create_option(
      name="num_devices",
      ty=int,
      docstring=
      "The number of devices attached to this input pipeline. This will be "
      "automatically set by MultiDeviceIterator.")
