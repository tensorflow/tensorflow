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

import glob
import json
import os
import sys
import tempfile
import threading
from absl.testing import parameterized
import numpy as np
import six

_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:  # pylint: disable=invalid-name
  _portpicker_import_error = _error
  portpicker = None

# pylint: disable=g-import-not-at-top
from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import parameter_server_strategy
from tensorflow.contrib.optimizer_v2 import adagrad
from tensorflow.core.protobuf import config_pb2
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
from tensorflow.python.feature_column import feature_column
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import server_lib

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

original_run_distribute_coordinator = dc.run_distribute_coordinator


# TODO(yuefengz): merge this method back to test_util.
def _create_local_cluster(num_workers,
                          num_ps,
                          has_eval=False,
                          protocol="grpc",
                          worker_config=None,
                          ps_config=None):
  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  if has_eval:
    cluster_dict["evaluator"] = ["localhost:%s" % portpicker.pick_unused_port()]

  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs,
          job_name="worker",
          protocol=protocol,
          task_index=ix,
          config=worker_config,
          start=True) for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs,
          job_name="ps",
          protocol=protocol,
          task_index=ix,
          config=ps_config,
          start=True) for ix in range(num_ps)
  ]
  if has_eval:
    evals = [
        server_lib.Server(
            cs,
            job_name="evaluator",
            protocol=protocol,
            task_index=0,
            config=worker_config,
            start=True)
    ]
  else:
    evals = []

  return workers, ps_servers, evals


def _create_in_process_cluster(num_workers, num_ps, has_eval=False):
  """Create an in-process cluster that consists of only standard server."""
  # Leave some memory for cuda runtime.
  if has_eval:
    gpu_mem_frac = 0.7 / (num_workers + 1)
  else:
    gpu_mem_frac = 0.7 / num_workers

  worker_config = config_pb2.ConfigProto()
  worker_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac

  # Enable collective ops which has no impact on non-collective ops.
  # TODO(yuefengz, tucker): removing this after we move the initialization of
  # collective mgr to the session level.
  worker_config.experimental.collective_group_leader = (
      "/job:worker/replica:0/task:0")

  ps_config = config_pb2.ConfigProto()
  ps_config.device_count["GPU"] = 0

  return _create_local_cluster(
      num_workers,
      num_ps=num_ps,
      has_eval=has_eval,
      worker_config=worker_config,
      ps_config=ps_config,
      protocol="grpc")


def _create_cluster_spec(has_chief=False,
                         num_workers=1,
                         num_ps=0,
                         has_eval=False):
  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type

  cluster_spec = {}
  if has_chief:
    cluster_spec[CHIEF] = ["localhost:%s" % portpicker.pick_unused_port()]
  if num_workers:
    cluster_spec[WORKER] = [
        "localhost:%s" % portpicker.pick_unused_port()
        for _ in range(num_workers)
    ]
  if num_ps:
    cluster_spec[PS] = [
        "localhost:%s" % portpicker.pick_unused_port() for _ in range(num_ps)
    ]
  if has_eval:
    cluster_spec[EVALUATOR] = ["localhost:%s" % portpicker.pick_unused_port()]
  return cluster_spec


def _bytes_to_str(maybe_bytes):
  if isinstance(maybe_bytes, six.string_types):
    return maybe_bytes
  else:
    return str(maybe_bytes, "utf-8")


def _strip_protocol(target):
  # cluster_spec expects "host:port" strings.
  if "//" in target:
    return target.split("//")[1]
  else:
    return target


class DistributeCoordinatorIntegrationTest(test.TestCase,
                                           parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    cls._workers, cls._ps, cls._evals = _create_in_process_cluster(
        num_workers=3, num_ps=2, has_eval=True)
    cls._cluster_spec = {
        "worker": [
            _strip_protocol(_bytes_to_str(w.target)) for w in cls._workers
        ],
        "ps": [_strip_protocol(_bytes_to_str(ps.target)) for ps in cls._ps],
        "evaluator": [
            _strip_protocol(_bytes_to_str(e.target)) for e in cls._evals
        ]
    }

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()
    self._event = threading.Event()
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
                     remote_cluster=None):
    estimator = self._get_estimator(train_distribute, eval_distribute,
                                    remote_cluster)

    input_dimension = LABEL_DIMENSION
    train_input_fn = self.dataset_input_fn(
        x={"x": DATA},
        y=DATA,
        batch_size=BATCH_SIZE // len(train_distribute.worker_devices),
        shuffle=True)
    if eval_distribute:
      eval_batch_size = BATCH_SIZE // len(eval_distribute.worker_devices)
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

    estimator_training.train_and_evaluate(
        estimator,
        estimator_training.TrainSpec(train_input_fn, max_steps=MAX_STEPS),
        estimator_training.EvalSpec(
            name=EVAL_NAME,
            input_fn=eval_input_fn,
            steps=None,
            exporters=self._get_exporter(EXPORTER_NAME, feature_columns),
            start_delay_secs=0,
            throttle_secs=1))
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

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          train_distribute_cls=[
              mirrored_strategy.MirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy
          ],
          eval_distribute_cls=[
              None, mirrored_strategy.MirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy
          ],
          required_gpus=1))
  def test_complete_flow_standalone_client(self, train_distribute_cls,
                                           eval_distribute_cls):
    try:
      train_distribute = train_distribute_cls(num_gpus=context.num_gpus())
    except TypeError:
      train_distribute = train_distribute_cls(num_gpus_per_worker=2)

    if eval_distribute_cls:
      eval_distribute = eval_distribute_cls()
    else:
      eval_distribute = None

    estimator = self._complete_flow(
        train_distribute, eval_distribute, remote_cluster=self._cluster_spec)
    self._inspect_train_and_eval_events(estimator)

  def _mock_run_distribute_coordinator(
      self,
      worker_fn,
      strategy,
      eval_fn,
      eval_strategy,
      mode=dc.CoordinatorMode.STANDALONE_CLIENT,
      cluster_spec=None,
      session_config=None):
    # Calls the origial `run_distribute_coordinator` method but gets task config
    # from environment variables and then signals the caller.
    task_type = None
    task_id = None
    if not cluster_spec:
      cluster_spec = None
      tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
      if not cluster_spec:
        cluster_spec = tf_config.get("cluster", {})
        task_env = tf_config.get("task", {})
        if task_env:
          task_type = task_env.get("type", task_type)
          task_id = int(task_env.get("index", task_id))
    self._event.set()
    original_run_distribute_coordinator(
        worker_fn,
        strategy,
        eval_fn,
        eval_strategy,
        mode=mode,
        cluster_spec=cluster_spec,
        task_type=task_type,
        task_id=task_id,
        session_config=session_config)

  def _task_thread(self, train_distribute, eval_distribute):
    with test.mock.patch.object(dc, "run_distribute_coordinator",
                                self._mock_run_distribute_coordinator):
      self._complete_flow(train_distribute, eval_distribute)

  def _run_task_in_thread(self, cluster_spec, task_type, task_id,
                          train_distribute, eval_distribute):
    if task_type:
      tf_config = {
          "cluster": cluster_spec,
          "task": {
              "type": task_type,
              "index": task_id
          }
      }
    else:
      tf_config = {
          "cluster": cluster_spec,
          "task": {
              "type": task_type,
              "index": task_id
          }
      }
    self._event.clear()
    t = threading.Thread(
        target=self._task_thread, args=(train_distribute, eval_distribute))
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(tf_config)}):
      t.start()
      self._event.wait()
    return t

  def _run_multiple_tasks_in_threads(self, cluster_spec, train_distribute,
                                     eval_distribute):
    threads = {}
    for task_type in cluster_spec.keys():
      threads[task_type] = []
      for task_id in range(len(cluster_spec[task_type])):
        t = self._run_task_in_thread(cluster_spec, task_type, task_id,
                                     train_distribute, eval_distribute)
        threads[task_type].append(t)
    return threads

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          train_distribute_cls=[
              parameter_server_strategy.ParameterServerStrategy,
          ],
          eval_distribute_cls=[
              None, mirrored_strategy.MirroredStrategy,
              parameter_server_strategy.ParameterServerStrategy
          ],
          required_gpus=1))
  def test_complete_flow_indepedent_worker_between_graph(
      self, train_distribute_cls, eval_distribute_cls):
    train_distribute = train_distribute_cls(
        num_gpus_per_worker=context.num_gpus())

    if eval_distribute_cls:
      eval_distribute = eval_distribute_cls()
    else:
      eval_distribute = None

    cluster_spec = _create_cluster_spec(num_workers=3, num_ps=2, has_eval=True)
    threads = self._run_multiple_tasks_in_threads(
        cluster_spec, train_distribute, eval_distribute)
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
          train_distribute_cls=[mirrored_strategy.MirroredStrategy],
          eval_distribute_cls=[None, mirrored_strategy.MirroredStrategy],
          required_gpus=1))
  def test_complete_flow_indepedent_worker_in_graph(self, train_distribute_cls,
                                                    eval_distribute_cls):
    train_distribute = train_distribute_cls(num_gpus=context.num_gpus())

    if eval_distribute_cls:
      eval_distribute = eval_distribute_cls()
    else:
      eval_distribute = None

    cluster_spec = _create_cluster_spec(num_workers=3, num_ps=2, has_eval=True)
    threads = self._run_multiple_tasks_in_threads(
        cluster_spec, train_distribute, eval_distribute)
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
              train_distribute=mirrored_strategy.MirroredStrategy(num_gpus=2)))

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
              train_distribute=mirrored_strategy.MirroredStrategy(num_gpus=2)))
      config_with_eval_distribute = run_config_lib.RunConfig(
          experimental_distribute=DistributeConfig(
              eval_distribute=mirrored_strategy.MirroredStrategy(num_gpus=2)))
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
              train_distribute=mirrored_strategy.MirroredStrategy(num_gpus=2)))
    self.assertFalse(dc_training.should_run_distribute_coordinator(config))

  def test_init_run_config_duplicate_distribute(self):
    with self.assertRaises(ValueError):
      run_config_lib.RunConfig(
          train_distribute=mirrored_strategy.MirroredStrategy(),
          experimental_distribute=DistributeConfig(
              train_distribute=mirrored_strategy.MirroredStrategy()))

    with self.assertRaises(ValueError):
      run_config_lib.RunConfig(
          eval_distribute=mirrored_strategy.MirroredStrategy(),
          experimental_distribute=DistributeConfig(
              eval_distribute=mirrored_strategy.MirroredStrategy()))

  def test_init_run_config_none_distribute_coordinator_mode(self):
    # We don't use distribute coordinator for local training.
    config = run_config_lib.RunConfig(
        train_distribute=mirrored_strategy.MirroredStrategy())
    dc_training.init_run_config(config, {})
    self.assertIsNone(config._distribute_coordinator_mode)

    # With a master in the cluster, don't run distribute coordinator.
    with test.mock.patch.dict("os.environ",
                              {"TF_CONFIG": json.dumps(TF_CONFIG_WITH_MASTER)}):
      config = run_config_lib.RunConfig(
          train_distribute=mirrored_strategy.MirroredStrategy())
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
          train_distribute=mirrored_strategy.MirroredStrategy())
    self.assertEqual(config._distribute_coordinator_mode,
                     dc.CoordinatorMode.INDEPENDENT_WORKER)

  def test_init_run_config_standalone_client(self):
    # When `train_distribute` is specified, TF_CONFIG is detected and
    # `experimental.remote_cluster` is set use distribute coordinator with
    # STANDALONE_CLIENT mode.
    config = run_config_lib.RunConfig(
        train_distribute=mirrored_strategy.MirroredStrategy(),
        experimental_distribute=DistributeConfig(
            remote_cluster={"chief": ["fake_worker"]}))
    self.assertEqual(config._distribute_coordinator_mode,
                     dc.CoordinatorMode.STANDALONE_CLIENT)


if __name__ == "__main__":
  with test.mock.patch.object(sys, "exit", os._exit):
    test.main()
