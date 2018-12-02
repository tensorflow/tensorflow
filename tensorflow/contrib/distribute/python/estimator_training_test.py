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
"""Tests that show Distribute Coordinator works with Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import glob
import json
import os
import sys
import tempfile
from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import collective_all_reduce_strategy
from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.contrib.distribute.python import parameter_server_strategy
from tensorflow.contrib.optimizer_v2 import adagrad
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import estimator_training as dc_training
from tensorflow.python.distribute.distribute_config import DistributeConfig
from tensorflow.python.eager import context
from tensorflow.python.estimator import exporter as exporter_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator import training as estimator_training
from tensorflow.python.estimator.canned import dnn_linear_combined
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export as export_lib
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import session_manager


BATCH_SIZE = 10
LABEL_DIMENSION = 2
DATA = np.linspace(
    0., 2., BATCH_SIZE * LABEL_DIMENSION, dtype=np.float32).reshape(
        BATCH_SIZE, LABEL_DIMENSION)
EVAL_NAME = "foo"
EXPORTER_NAME = "saved_model_exporter"
MAX_STEPS = 10

CHIEF = dc._TaskType.CHIEF
EVALUATOR = dc._TaskType.EVALUATOR
WORKER = dc._TaskType.WORKER
PS = dc._TaskType.PS

original_run_std_server = dc._run_std_server


class DistributeCoordinatorIntegrationTest(
    multi_worker_test_base.IndependentWorkerTestBase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    super(DistributeCoordinatorIntegrationTest, cls).setUpClass()
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=2, has_eval=True)

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()
    super(DistributeCoordinatorIntegrationTest, self).setUp()

  def dataset_input_fn(self, x, y, batch_size, shuffle):

    def input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      if shuffle:
        dataset = dataset.shuffle(batch_size)
      dataset = dataset.repeat(100).batch(batch_size)
      return dataset

    return input_fn

  def _get_exporter(self, name, fc):
    feature_spec = feature_column.make_parse_example_spec(fc)
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    return exporter_lib.LatestExporter(
        name, serving_input_receiver_fn=serving_input_receiver_fn)

  def _extract_loss_and_global_step(self, event_folder):
    """Returns the loss and global step in last event."""
    event_paths = glob.glob(os.path.join(event_folder, "events*"))
    self.assertNotEmpty(
        event_paths, msg="Event file not found in dir %s" % event_folder)

    loss = None
    global_step_count = None

    for e in summary_iterator.summary_iterator(event_paths[-1]):
      current_loss = None
      for v in e.summary.value:
        if v.tag == "loss":
          current_loss = v.simple_value

      # If loss is not found, global step is meaningless.
      if current_loss is None:
        continue

      current_global_step = e.step
      if global_step_count is None or current_global_step > global_step_count:
        global_step_count = current_global_step
        loss = current_loss

    return (loss, global_step_count)

  def _get_estimator(self,
                     train_distribute,
                     eval_distribute,
                     remote_cluster=None):
    input_dimension = LABEL_DIMENSION
    linear_feature_columns = [
        feature_column.numeric_column("x", shape=(input_dimension,))
    ]
    dnn_feature_columns = [
        feature_column.numeric_column("x", shape=(input_dimension,))
    ]

    return dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=linear_feature_columns,
        dnn_hidden_units=(2, 2),
        dnn_feature_columns=dnn_feature_columns,
        label_dimension=LABEL_DIMENSION,
        model_dir=self._model_dir,
        dnn_optimizer=adagrad.AdagradOptimizer(0.001),
        linear_optimizer=adagrad.AdagradOptimizer(0.001),
        config=run_config_lib.RunConfig(
            experimental_distribute=DistributeConfig(
                train_distribute=train_distribute,
                eval_distribute=eval_distribute,
                remote_cluster=remote_cluster)))

  def _complete_flow(self,
                     train_distribute,
                     eval_distribute,
                     remote_cluster=None,
                     use_train_and_evaluate=True):
    estimator = self._get_estimator(train_distribute, eval_distribute,
                                    remote_cluster)

    input_dimension = LABEL_DIMENSION
    train_input_fn = self.dataset_input_fn(
        x={"x": DATA},
        y=DATA,
        batch_size=BATCH_SIZE // train_distribute.num_replicas_in_sync,
        shuffle=True)
    if eval_distribute:
      eval_batch_size = BATCH_SIZE // eval_distribute.num_replicas_in_sync
    else:
      eval_batch_size = BATCH_SIZE
    eval_input_fn = self.dataset_input_fn(
        x={"x": DATA}, y=DATA, batch_size=eval_batch_size, shuffle=False)

    linear_feature_columns = [
        feature_column.numeric_column("x", shape=(input_dimension,))
    ]
    dnn_feature_columns = [
        feature_column.numeric_column("x", shape=(input_dimension,))
    ]
    feature_columns = linear_feature_columns + dnn_feature_columns

    eval_spec = estimator_training.EvalSpec(
        name=EVAL_NAME,
        input_fn=eval_input_fn,
        steps=None,
        exporters=self._get_exporter(EXPORTER_NAME, feature_columns),
        start_delay_secs=0,
        throttle_secs=1)

    if use_train_and_evaluate:
      estimator_training.train_and_evaluate(
          estimator,
          estimator_training.TrainSpec(train_input_fn, max_steps=MAX_STEPS),
          eval_spec)
    else:
      estimator.train(train_input_fn, max_steps=MAX_STEPS)

      latest_ckpt_path = estimator.latest_checkpoint()
      metrics = estimator.evaluate(eval_input_fn,
                                   checkpoint_path=latest_ckpt_path,
                                   name=EVAL_NAME)

      # Export the eval result to files.
      eval_result = estimator_training._EvalResult(
          status=estimator_training._EvalStatus.EVALUATED,
          metrics=metrics,
          checkpoint_path=latest_ckpt_path)
      evaluator = estimator_training._TrainingExecutor._Evaluator(estimator,
                                                                  eval_spec,
                                                                  None)
      evaluator._export_eval_result(eval_result, True)

    return estimator

  def _inspect_train_and_eval_events(self, estimator):
    # Make sure nothing is stuck in limbo.
    writer_cache.FileWriterCache.clear()

    # Examine the training events. Use a range to check global step to avoid
    # flakyness due to global step race condition.
    training_loss, _ = self._extract_loss_and_global_step(self._model_dir)
    self.assertIsNotNone(training_loss)

    # Examine the eval events. The global step should be accurate.
    eval_dir = os.path.join(self._model_dir, "eval_" + EVAL_NAME)
    eval_loss, eval_global_step = self._extract_loss_and_global_step(
        event_folder=eval_dir)
    self.assertIsNotNone(eval_loss)
    self.assertGreaterEqual(eval_global_step, MAX_STEPS)

    # Examine the export folder.
    export_dir = os.path.join(
        os.path.join(self._model_dir, "export"), EXPORTER_NAME)
    self.assertTrue(gfile.Exists(export_dir))

    # Examine the ckpt for predict.
    def predict_input_fn():
      return dataset_ops.Dataset.from_tensor_slices({
          "x": DATA
      }).batch(BATCH_SIZE)

    predicted_proba = np.array([
        x[prediction_keys.PredictionKeys.PREDICTIONS]
        for x in estimator.predict(predict_input_fn)
    ])
    self.assertAllEqual((BATCH_SIZE, LABEL_DIMENSION), predicted_proba.shape)

  def _get_strategy_object(self, strategy_cls):
    if strategy_cls == mirrored_strategy.CoreMirroredStrategy:
      return strategy_cls(mirrored_strategy.all_local_devices())
    else:
      return strategy_cls(num_gpus_per_worker=context.num_gpus())

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          train_distribute_cls=[
              collective_all_reduce_strategy.CollectiveAllReduceStrategy,
              mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy
          ],
          eval_distribute_cls=[
              None,
              mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy,
          ],
          required_gpus=[0, 1]))
  def test_complete_flow_standalone_client(self, train_distribute_cls,
                                           eval_distribute_cls):
    train_distribute = self._get_strategy_object(train_distribute_cls)

    if eval_distribute_cls:
      eval_distribute = self._get_strategy_object(eval_distribute_cls)
    else:
      eval_distribute = None

    cluster_spec = copy.deepcopy(self._cluster_spec)
    if (train_distribute_cls !=
        parameter_server_strategy.ParameterServerStrategy):
      cluster_spec.pop("ps", None)
    estimator = self._complete_flow(train_distribute, eval_distribute,
                                    cluster_spec)
    self._inspect_train_and_eval_events(estimator)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          train_distribute_cls=[
              mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy,
          ],
          eval_distribute_cls=[
              None,
              mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy,
          ],
          required_gpus=[0, 1]))
  def test_estimator_standalone_client(self, train_distribute_cls,
                                       eval_distribute_cls):
    train_distribute = self._get_strategy_object(train_distribute_cls)

    if eval_distribute_cls:
      eval_distribute = self._get_strategy_object(eval_distribute_cls)
    else:
      eval_distribute = None

    # We use the whole cluster for evaluation.
    cluster = copy.deepcopy(self._cluster_spec)
    cluster.pop("evaluator", None)

    estimator = self._complete_flow(
        train_distribute, eval_distribute, remote_cluster=cluster,
        use_train_and_evaluate=False)
    self._inspect_train_and_eval_events(estimator)

  def _mock_run_std_server(self, *args, **kwargs):
    ret = original_run_std_server(*args, **kwargs)
    # Wait for all std servers to be brought up in order to reduce the chance of
    # remote sessions taking local ports that have been assigned to std servers.
    self._barrier.wait()
    return ret

  def _independent_worker_fn(
      self,
      train_distribute,
      eval_distribute,
  ):
    with test.mock.patch.object(dc, "_run_std_server",
                                self._mock_run_std_server):
      self._complete_flow(train_distribute, eval_distribute)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          train_distribute_cls=[
              collective_all_reduce_strategy.CollectiveAllReduceStrategy,
              parameter_server_strategy.ParameterServerStrategy,
          ],
          eval_distribute_cls=[
              None, mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy,
          ],
          required_gpus=[0, 1]))
  def test_complete_flow_indepedent_worker_between_graph(
      self, train_distribute_cls, eval_distribute_cls):
    if (context.num_gpus() < 2 and eval_distribute_cls ==
        collective_all_reduce_strategy.CollectiveAllReduceStrategy):
      self.skipTest("`CollectiveAllReduceStrategy` needs at least two towers.")

    train_distribute = self._get_strategy_object(train_distribute_cls)

    if eval_distribute_cls:
      eval_distribute = self._get_strategy_object(eval_distribute_cls)
    else:
      eval_distribute = None

    if (train_distribute_cls == parameter_server_strategy
        .ParameterServerStrategy):
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          num_workers=3, num_ps=2, has_eval=True)
      # 3 workers, 2 ps and 1 evaluator.
      self._barrier = dc._Barrier(6)
    else:
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          num_workers=3, num_ps=0, has_eval=True)
      # 3 workers and 1 evaluator.
      self._barrier = dc._Barrier(4)

    threads = self.run_multiple_tasks_in_threads(self._independent_worker_fn,
                                                 cluster_spec, train_distribute,
                                                 eval_distribute)
    for task_type, ts in threads.items():
      if task_type == PS:
        continue
      for t in ts:
        t.join()

    estimator = self._get_estimator(train_distribute, eval_distribute)
    self._inspect_train_and_eval_events(estimator)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          train_distribute_cls=[
              mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy
          ],
          eval_distribute_cls=[
              None,
              mirrored_strategy.MirroredStrategy,
              mirrored_strategy.CoreMirroredStrategy
          ],
          required_gpus=[0, 1]))
  def test_complete_flow_indepedent_worker_in_graph(self, train_distribute_cls,
                                                    eval_distribute_cls):
    train_distribute = self._get_strategy_object(train_distribute_cls)

    if eval_distribute_cls:
      eval_distribute = self._get_strategy_object(eval_distribute_cls)
    else:
      eval_distribute = None

    cluster_spec = multi_worker_test_base.create_cluster_spec(
        num_workers=3, num_ps=0, has_eval=True)
    # 3 workers and 1 evaluator.
    self._barrier = dc._Barrier(4)
    threads = self.run_multiple_tasks_in_threads(self._independent_worker_fn,
                                                 cluster_spec, train_distribute,
                                                 eval_distribute)
    threads[WORKER][0].join()
    threads[EVALUATOR][0].join()

    estimator = self._get_estimator(train_distribute, eval_distribute)
    self._inspect_train_and_eval_events(estimator)


TF_CONFIG_WITH_CHIEF = {
    "cluster": {
        "chief": ["fake_chief"],
    },
    "task": {
        "type": "chief",
        "index": 0
    }
}

TF_CONFIG_WITH_MASTER = {
    "cluster": {
        "master": ["fake_master"],
    },
    "task": {
        "type": "master",
        "index": 0
    }
}

TF_CONFIG_WITHOUT_TASK = {"cluster": {"chief": ["fake_worker"]}}


class RunConfigTest(test.TestCase):

  def test_previously_unexpected_cluster_spec(self):
    with test.mock.patch.dict(
        "os.environ", {"TF_CONFIG": json.dumps(TF_CONFIG_WITHOUT_TASK)}):
      run_config_lib.RunConfig(
          experimental_distribute=DistributeConfig(
              train_distribute=mirrored_strategy.CoreMirroredStrategy(
                  ["/device:GPU:0", "/device:GPU:1"])))

  def test_should_run_distribute_coordinator(self):
    """Tests that should_run_distribute_coordinator return a correct value."""
    # We don't use distribute coordinator for local training.
    self.assertFalse(
        dc_training.should_run_distribute_coordinator(
            run_config_lib.RunConfig()))

    # When `train_distribute` is not specified, don't use distribute
    # coordinator.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_CHIEF)}):
      self.assertFalse(
          dc_training.should_run_distribute_coordinator(
              run_config_lib.RunConfig()))

    # When `train_distribute` is specified and TF_CONFIG is detected, use
    # distribute coordinator.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_CHIEF)}):
      config_with_train_distribute = run_config_lib.RunConfig(
          experimental_distribute=DistributeConfig(
              train_distribute=mirrored_strategy.CoreMirroredStrategy(
                  ["/device:GPU:0", "/device:GPU:1"])))
      config_with_eval_distribute = run_config_lib.RunConfig(
          experimental_distribute=DistributeConfig(
              eval_distribute=mirrored_strategy.CoreMirroredStrategy(
                  ["/device:GPU:0", "/device:GPU:1"])))
    self.assertTrue(
        dc_training.should_run_distribute_coordinator(
            config_with_train_distribute))
    self.assertFalse(
        dc_training.should_run_distribute_coordinator(
            config_with_eval_distribute))

    # With a master in the cluster, don't run distribute coordinator.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_MASTER)}):
      config = run_config_lib.RunConfig(
          experimental_distribute=DistributeConfig(
              train_distribute=mirrored_strategy.CoreMirroredStrategy(
                  ["/device:GPU:0", "/device:GPU:1"])))
    self.assertFalse(dc_training.should_run_distribute_coordinator(config))

  def test_init_run_config_duplicate_distribute(self):
    with self.assertRaises(ValueError):
      run_config_lib.RunConfig(
          train_distribute=mirrored_strategy.CoreMirroredStrategy(),
          experimental_distribute=DistributeConfig(
              train_distribute=mirrored_strategy.CoreMirroredStrategy()))

    with self.assertRaises(ValueError):
      run_config_lib.RunConfig(
          eval_distribute=mirrored_strategy.CoreMirroredStrategy(),
          experimental_distribute=DistributeConfig(
              eval_distribute=mirrored_strategy.CoreMirroredStrategy()))

  def test_init_run_config_none_distribute_coordinator_mode(self):
    # We don't use distribute coordinator for local training.
    config = run_config_lib.RunConfig(
        train_distribute=mirrored_strategy.CoreMirroredStrategy())
    dc_training.init_run_config(config, {})
    self.assertIsNone(config._distribute_coordinator_mode)

    # With a master in the cluster, don't run distribute coordinator.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_MASTER)}):
      config = run_config_lib.RunConfig(
          train_distribute=mirrored_strategy.CoreMirroredStrategy())
      self.assertIsNone(config._distribute_coordinator_mode)

    # When `train_distribute` is not specified, don't use distribute
    # coordinator.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_CHIEF)}):
      config = run_config_lib.RunConfig()
      self.assertFalse(hasattr(config, "_distribute_coordinator_mode"))

  def test_init_run_config_independent_worker(self):
    # When `train_distribute` is specified and TF_CONFIG is detected, use
    # distribute coordinator with INDEPENDENT_WORKER mode.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_CHIEF)}):
      config = run_config_lib.RunConfig(
          train_distribute=mirrored_strategy.CoreMirroredStrategy())
    self.assertEqual(config._distribute_coordinator_mode,
                     dc.CoordinatorMode.INDEPENDENT_WORKER)

  def test_init_run_config_standalone_client(self):
    # When `train_distribute` is specified, TF_CONFIG is detected and
    # `experimental.remote_cluster` is set use distribute coordinator with
    # STANDALONE_CLIENT mode.
    config = run_config_lib.RunConfig(
        train_distribute=mirrored_strategy.CoreMirroredStrategy(),
        experimental_distribute=DistributeConfig(
            remote_cluster={"chief": ["fake_worker"]}))
    self.assertEqual(config._distribute_coordinator_mode,
                     dc.CoordinatorMode.STANDALONE_CLIENT)


if __name__ == "__main__":
  # Reduce `recovery_wait_secs` from 30 seconds so the test completes quickly.
  orig_init = session_manager.SessionManager.__init__

  def new_init(*args, **kwargs):
    kwargs.pop("recovery_wait_secs", None)
    kwargs["recovery_wait_secs"] = 0.5
    orig_init(*args, **kwargs)

  session_manager.SessionManager.__init__ = new_init

  with test.mock.patch.object(sys, "exit", os._exit):
    test.main()
