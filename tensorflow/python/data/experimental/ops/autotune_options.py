# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""API for controlling performance auto-tuning in `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.AutotuneOptions")
class AutotuneOptions(options.OptionsBase):
  """Represents options for performance auto-tuning in
  dataset operations.

  You can set the auto-tune options of a dataset through the
  `autotune` property of `tf.data.Options`; the property is
  an instance of `tf.data.AutotuneOptions`.

  ```python
  options = tf.data.Options()
  options.autotune.enabled = True
  options.autotune.cpu_budget = 8
  dataset = dataset.with_options(options)
  ```
  """
  enabled = options.create_option(
      name="enabled",
      ty=bool,
      docstring=
      "Whether to automatically tune performance knobs. If None, defaults to "
      "True.")

  experimental_autotune_buffers = options.create_option(
      name="experimental_autotune_buffers",
      ty=bool,
      docstring=
      "When autotuning is enabled (through `enabled`), determines whether to "
      "also autotune buffer sizes for datasets with parallelism. If None,"
      " defaults to False.")

  cpu_budget = options.create_option(
      name="cpu_budget",
      ty=int,
      docstring=
      "When autotuning is enabled (through `autotune`), determines the CPU "
      "budget to use. Values greater than the number of schedulable CPU cores "
      "are allowed but may result in CPU contention. If None, defaults to the "
      "number of schedulable CPU cores.")

  ram_budget = options.create_option(
      name="ram_budget",
      ty=int,
      docstring=
      "When autotuning is enabled (through `autotune`), determines the RAM "
      "budget to use. Values greater than the available RAM in bytes may "
      "result in OOM. If None, defaults to half of the available RAM in bytes.")

  def _to_proto(self):
    pb = dataset_options_pb2.AutotuneOptions()
    if self.enabled is not None:
      pb.enabled = self.enabled
    if self.experimental_autotune_buffers is not None:
      pb.autotune_buffers = self.experimental_autotune_buffers
    if self.cpu_budget is not None:
      pb.cpu_budget = self.cpu_budget
    if self.ram_budget is not None:
      pb.ram_budget = self.ram_budget
    return pb

  def _from_proto(self, pb):
    if pb.WhichOneof("optional_enabled") is not None:
      self.enabled = pb.enabled
    if pb.WhichOneof("optional_autotune_buffers") is not None:
      self.experimental_autotune_buffers = pb.autotune_buffers
    if pb.WhichOneof("optional_cpu_budget") is not None:
      self.cpu_budget = pb.cpu_budget
    if pb.WhichOneof("optional_ram_budget") is not None:
      self.ram_budget = pb.ram_budget

  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
