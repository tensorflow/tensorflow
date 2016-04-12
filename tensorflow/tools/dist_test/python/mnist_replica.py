# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --worker_index. There should be exactly one invocation with
--worker_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multpile worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("worker_index", 0,
                     "Worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_workers", None,
                     "Total number of workers (must be >= 1)")
flags.DEFINE_integer("num_parameter_servers", 2,
                     "Total number of parameter servers (must be >= 1)")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before paramter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("grpc_port", 2222,
                     "TensorFlow GRPC port")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_string("worker_grpc_url", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workersare aggregated "
                     "before applied to avoid stale gradients")
FLAGS = flags.FLAGS


IMAGE_PIXELS = 28

PARAM_SERVER_PREFIX = "tf-ps"  # Prefix of the parameter servers' domain names
WORKER_PREFIX = "tf-worker"  # Prefix of the workers' domain names


def get_device_setter(num_parameter_servers, num_workers):
  """Get a device setter given number of servers in the cluster.

  Given the numbers of parameter servers and workers, construct a device
  setter object using ClusterSpec.

  Args:
    num_parameter_servers: Number of parameter servers
    num_workers: Number of workers

  Returns:
    Device setter object.
  """

  ps_spec = []
  for j in range(num_parameter_servers):
    ps_spec.append("%s%d:%d" % (PARAM_SERVER_PREFIX, j, FLAGS.grpc_port))

  worker_spec = []
  for k in range(num_workers):
    worker_spec.append("%s%d:%d" % (WORKER_PREFIX, k, FLAGS.grpc_port))

  cluster_spec = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  # Get device setter from the cluster spec
  return tf.train.replica_device_setter(cluster=cluster_spec)


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)

  print("Worker GRPC URL: %s" % FLAGS.worker_grpc_url)
  print("Worker index = %d" % FLAGS.worker_index)
  print("Number of workers = %d" % FLAGS.num_workers)

  # Sanity check on the number of workers and the worker index
  if FLAGS.worker_index >= FLAGS.num_workers:
    raise ValueError("Worker index %d exceeds number of workers %d " %
                     (FLAGS.worker_index, FLAGS.num_workers))

  # Sanity check on the number of parameter servers
  if FLAGS.num_parameter_servers <= 0:
    raise ValueError("Invalid num_parameter_servers value: %d" %
                     FLAGS.num_parameter_servers)

  is_chief = (FLAGS.worker_index == 0)

  if FLAGS.sync_replicas:
    if FLAGS.replicas_to_aggregate is None:
      replicas_to_aggregate = FLAGS.num_workers
    else:
      replicas_to_aggregate = FLAGS.replicas_to_aggregate

  # Construct device setter object
  device_setter = get_device_setter(FLAGS.num_parameter_servers,
                                    FLAGS.num_workers)

  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  with tf.device(device_setter):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                            stddev=1.0 / IMAGE_PIXELS), name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal([FLAGS.hidden_units, 10],
                            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.worker_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ *
                                   tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    if FLAGS.sync_replicas:
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=FLAGS.num_workers,
          replica_id=FLAGS.worker_index,
          name="mnist_sync_replicas")

    train_step = opt.minimize(cross_entropy,
                              global_step=global_step)

    if FLAGS.sync_replicas and is_chief:
      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      init_tokens_op = opt.get_init_tokens_op()

    init_op = tf.initialize_all_variables()
    train_dir = tempfile.mkdtemp()
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=train_dir,
                             init_op=init_op,
                             recovery_wait_secs=1,
                             global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.worker_index])

    # The chief worker (worker_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.worker_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.worker_index)

    sess = sv.prepare_or_wait_for_session(FLAGS.worker_grpc_url,
                                          config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.worker_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op
      print("Starting chief queue runner and running init_tokens_op")
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_tokens_op)

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs,
                    y_: batch_ys}

      _, step = sess.run([train_step, global_step], feed_dict=train_feed)
      local_step += 1

      now = time.time()
      print("%f: Worker %d: training step %d done (global step: %d)" %
            (now, FLAGS.worker_index, local_step, step))

      if step >= FLAGS.train_steps:
        break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    # Validation feed
    val_feed = {x: mnist.validation.images,
                y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()
