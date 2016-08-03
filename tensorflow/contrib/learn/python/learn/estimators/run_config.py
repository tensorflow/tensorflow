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

"""Run Config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import ConfigProto
from tensorflow.python import GPUOptions


class RunConfig(object):
  """This class specifies the specific configurations for the run."""

  # TODO(wicke): Move options out once functionality is covered by monitors
  def __init__(self,
               master='',
               task=0,
               num_ps_replicas=0,
               num_cores=4,
               log_device_placement=False,
               gpu_memory_fraction=1,
               tf_random_seed=42,
               save_summary_steps=100,
               save_checkpoints_secs=60,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000):
    """Constructor.

    Args:
      master: TensorFlow master. Empty string (the default) for local.
      task: Task id of the replica running the training (default: 0).
      num_ps_replicas: Number of parameter server tasks to use (default: 0).
      num_cores: Number of cores to be used (default: 4).
      log_device_placement: Log the op placement to devices (default: False).
      gpu_memory_fraction: Fraction of GPU memory used by the process on
        each GPU uniformly on the same machine.
      tf_random_seed: Random seed for TensorFlow initializers.
        Setting this value allows consistency between reruns.
      save_summary_steps: Save summaries every this many steps.
      save_checkpoints_secs: Save checkpoints every this many seconds.
      keep_checkpoint_max: The maximum number of recent checkpoint files to
        keep. As new files are created, older files are deleted. If None or 0,
        all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
        checkpoint files are kept.)
      keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved. The default value of 10,000 hours effectively disables
        the feature.
    """
    self.master = master
    self.task = task
    self.num_ps_replicas = num_ps_replicas
    gpu_options = GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    self.tf_config = ConfigProto(log_device_placement=log_device_placement,
                                 inter_op_parallelism_threads=num_cores,
                                 intra_op_parallelism_threads=num_cores,
                                 gpu_options=gpu_options)
    self.tf_random_seed = tf_random_seed
    self.save_summary_steps = save_summary_steps
    self.save_checkpoints_secs = save_checkpoints_secs
    self.keep_checkpoint_max = keep_checkpoint_max
    self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
