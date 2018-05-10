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
r"""Train a ConvNet on MNIST using K-FAC.

Distributed training with sync replicas optimizer. See
`convnet.train_mnist_distributed_sync_replicas` for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import tensorflow as tf

from tensorflow.contrib.kfac.examples import convnet

FLAGS = flags.FLAGS
flags.DEFINE_integer("task", -1, "Task identifier")
flags.DEFINE_string("data_dir", "/tmp/mnist", "local mnist dir")
flags.DEFINE_string(
    "cov_inv_op_strategy", "chief_worker",
    "In dist training mode run the cov, inv ops on chief or dedicated workers."
)
flags.DEFINE_string("master", "local", "Session master.")
flags.DEFINE_integer("ps_tasks", 2,
                     "Number of tasks in the parameter server job.")
flags.DEFINE_integer("replicas_to_aggregate", 5,
                     "Number of replicas to aggregate.")
flags.DEFINE_integer("worker_replicas", 5, "Number of replicas in worker job.")
flags.DEFINE_integer("num_epochs", None, "Number of epochs.")


def _is_chief():
  """Determines whether a job is the chief worker."""
  if "chief_worker" in FLAGS.brain_jobs:
    return FLAGS.brain_job_name == "chief_worker"
  else:
    return FLAGS.task == 0


def main(unused_argv):
  _ = unused_argv
  convnet.train_mnist_distributed_sync_replicas(
      FLAGS.task, _is_chief(), FLAGS.worker_replicas, FLAGS.ps_tasks,
      FLAGS.master, FLAGS.data_dir, FLAGS.num_epochs, FLAGS.cov_inv_op_strategy)

if __name__ == "__main__":
  tf.app.run(main=main)
