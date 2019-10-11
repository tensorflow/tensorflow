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

from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.DistributeOptions")
class DistributeOptions(options.OptionsBase):
  """Represents options for distributed data processing.

  You can set the distribution options of a dataset through the
  `experimental_distribute` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.DistributeOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_distribute.auto_shard = False
  dataset = dataset.with_options(options)
  ```
  """

  auto_shard = options.create_option(
      name="auto_shard",
      ty=bool,
      docstring=
      "Whether the dataset should be automatically sharded when processed"
      "in a distributed fashion. This is applicable when using Keras with "
      "multi-worker/TPU distribution strategy, and by "
      "using strategy.experimental_distribute_dataset(). In other cases, this "
      "option does nothing. If None, defaults to True.",
      default_factory=lambda: True)

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
