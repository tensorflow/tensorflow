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
"""Test async checkpointing."""

import os

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.tpu import async_checkpoint
from tensorflow.python.tpu import tpu_config
from tensorflow.python.tpu import tpu_estimator
from tensorflow.python.tpu import tpu_optimizer
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import training
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'TPU to use in this test.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string(
    'model_dir',
    os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'),
    'GCS path to store model and checkpoints.')


def _get_checkpoint_metrics_counts() -> (int, int):
  """Get the count for recorded sync and async checkpoint write durations."""
  def get_count(method):
    proto_bytes = method(api_label=async_checkpoint._ASYNC_CHECKPOINT_V1)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return int(histogram_proto.num)
  return get_count(metrics.GetCheckpointWriteDurations), get_count(
      metrics.GetAsyncCheckpointWriteDurations)


def input_fn(params):
  """Return a dataset of source and target sequences for training."""
  return (constant_op.constant(
      np.random.randn(params['batch_size'], 1000), dtype=dtypes.float32),
          constant_op.constant(
              np.random.randint(0, 10, params['batch_size']),
              dtype=dtypes.int32))


def model_fn(features, labels, mode, params):
  del params  # unused
  with variable_scope.variable_scope('m', reuse=variable_scope.AUTO_REUSE):
    w = variable_scope.get_variable('W', shape=[1000, 10])
  logits = math_ops.matmul(features, w)
  loss = losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == model_fn_lib.ModeKeys.TRAIN:
    optimizer = training.RMSPropOptimizer(learning_rate=0.01)
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, training.get_global_step())
    return tpu_estimator.TPUEstimatorSpec(
        mode=model_fn_lib.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
    )
  elif mode == model_fn_lib.ModeKeys.EVAL:

    def metric_fn(labels, logits):
      labels = math_ops.cast(labels, dtypes.int64)
      logging.info('LABELS %s %s', labels, logits)
      return {
          'recall@1': metrics_lib.recall_at_k(labels, logits, 1),
          'recall@5': metrics_lib.recall_at_k(labels, logits, 5),
      }

    loss = losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    eval_metrics = (metric_fn, [labels, logits])
    return tpu_estimator.TPUEstimatorSpec(
        mode=model_fn_lib.ModeKeys.EVAL, loss=loss, eval_metrics=eval_metrics)


class AsyncCheckpointingTest(test.TestCase):

  def testAsyncCheckpointHookEnabled(self):
    resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)

    checkpoint_interval = 5
    config = tpu_config.RunConfig(
        master=resolver.master(),
        model_dir=os.path.join(FLAGS.model_dir, 'runconfig'),
        save_checkpoints_steps=1000,
        keep_checkpoint_max=11,  # off by one
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=checkpoint_interval,))

    estimator = tpu_estimator.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=32,
        eval_batch_size=32,
        predict_batch_size=1,
        params={},
    )

    max_steps = 100
    mock_listener = test.mock.create_autospec(
        basic_session_run_hooks.CheckpointSaverListener)
    estimator.train(
        input_fn=input_fn,
        max_steps=max_steps,
        hooks=[
            async_checkpoint.AsyncCheckpointSaverHook(
                FLAGS.model_dir,
                save_steps=checkpoint_interval,
                listeners=[mock_listener])
        ])

    current_step = estimator_lib._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)  # pylint: disable=protected-access

    # TODO(power) -- identify a better way to count the number of checkpoints.
    checkpoints = file_io.get_matching_files(
        FLAGS.model_dir + '/model.ckpt*.meta')
    checkpoint_count = len(checkpoints)
    logging.info('Found %d checkpoints: %s', checkpoint_count, checkpoints)
    self.assertLessEqual(checkpoint_count, 10)
    self.assertEqual(current_step, max_steps)
    mock_listener.before_save.assert_called()
    mock_listener.after_save.assert_called()

    # save called by hook in `after_create_session` and every `after_run`
    num_save_calls = 1 + max_steps // checkpoint_interval
    sync_count, async_count = _get_checkpoint_metrics_counts()
    # save might be called one extra time in `end` hook based on timing of
    # `_last_checkpoint_step` update in the final `after_run` call
    self.assertIn(sync_count, [num_save_calls, num_save_calls + 1])
    self.assertLessEqual(async_count, num_save_calls)

  def testAsyncCheckpointHookWithoutListeners(self):
    resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)

    checkpoint_interval = 5
    keep_checkpoint_max = 10
    config = tpu_config.RunConfig(
        master=resolver.master(),
        model_dir=os.path.join(FLAGS.model_dir, 'runconfig'),
        save_checkpoints_steps=1000,
        keep_checkpoint_max=keep_checkpoint_max+1,  # off by one
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=checkpoint_interval,))

    estimator = tpu_estimator.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=32,
        eval_batch_size=32,
        predict_batch_size=1,
        params={},
    )

    max_steps = 100
    estimator.train(
        input_fn=input_fn,
        max_steps=max_steps,
        hooks=[
            async_checkpoint.AsyncCheckpointSaverHook(
                FLAGS.model_dir,
                save_steps=checkpoint_interval)
        ])

    current_step = estimator_lib._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)  # pylint: disable=protected-access

    # TODO(power) -- identify a better way to count the number of checkpoints.
    checkpoints = file_io.get_matching_files(
        FLAGS.model_dir + '/model.ckpt*.meta')
    checkpoint_count = len(checkpoints)
    logging.info('Found %d checkpoints: %s', checkpoint_count, checkpoints)
    self.assertLessEqual(checkpoint_count, keep_checkpoint_max)
    self.assertEqual(current_step, max_steps)

    # save called by hook in `after_create_session` and every `after_run`
    num_save_calls = 1 + max_steps // checkpoint_interval
    sync_count, async_count = _get_checkpoint_metrics_counts()
    # save might be called one extra time in `end` hook based on timing of
    # `_last_checkpoint_step` update in the final `after_run` call
    self.assertIn(sync_count, [num_save_calls, num_save_calls + 1])
    self.assertLessEqual(async_count, num_save_calls)


if __name__ == '__main__':
  v2_compat.disable_v2_behavior()
  test.main()
