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
"""Tests of `worker_training_state.py` utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import multiprocessing as mp
import os
import sys
import tempfile

from absl.testing import parameterized
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class ModelCheckpointTest(test_base.IndependentWorkerTestBase,
                          parameterized.TestCase):

  @ds_combinations.generate(
      combinations.combine(mode=['graph'],
                           required_gpus=[0, 1],
                           file_format=['h5', 'tf'],
                           save_weights_only=[True, False]))
  def testCheckpointExists(self, file_format, save_weights_only):
    with self.cached_session():
      train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(64, 2)
      model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))
      saving_dir = self.get_temp_dir()
      saving_filepath = os.path.join(saving_dir, 'checkpoint.' + file_format)
      callbacks_list = [
          callbacks.ModelCheckpoint(filepath=saving_filepath,
                                    save_weights_only=save_weights_only)
      ]
      self.assertFalse(file_io.file_exists_v2(saving_filepath))
      model.fit(
          x=train_ds, epochs=2, steps_per_epoch=2, callbacks=callbacks_list)
      tf_saved_model_exists = file_io.file_exists_v2(saving_filepath)
      tf_weights_only_checkpoint_exists = file_io.file_exists_v2(
          saving_filepath + '.index')
      self.assertTrue(tf_saved_model_exists or
                      tf_weights_only_checkpoint_exists)


def testWorkerTrainingState(cluster_spec, task_id, backup_dir, create_barrier):
  #Set TFCONFIG
  tf_config = {
      'cluster': cluster_spec,
      'task': {
          'type': "worker",
          'index': task_id
      }
  }
  os.environ["TF_CONFIG"] = json.dumps(tf_config)
  strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
  train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(64, 2)
  with strategy.scope():
    model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))

  model._training_state = (worker_training_state.WorkerTrainingState(
      model, backup_dir))
  model._training_state.back_up(0)

  len_backup = len(os.listdir(backup_dir))
  assert len_backup > 0, "%d" % (len_backup)
  create_barrier.wait()

  model._training_state.restore()
  model._training_state.delete_backup()

  len_backup = len(os.listdir(backup_dir))
  if task_id == 0:
    assert len_backup == 0, "%d" % (len_backup)

  return


class WorkerTrainingStateTest(test.TestCase, parameterized.TestCase):

  def create_cluster_spec(self, num_workers):
    worker_ports = [test_base.pick_unused_port() for _ in range(num_workers)]
    cluster_dict = {}
    if num_workers > 0:
      cluster_dict['worker'] = ['localhost:%s' % port for port in worker_ports]
    self._cluster_spec = cluster_dict

  def testWorkerTrainingState(self):
    try:
      num_workers = 3
      self._cluster_spec = test_base.create_cluster_spec(num_workers=num_workers)
      pool = mp.Pool(num_workers)
      
      backup_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
      with mp.Manager() as man:
        create_barrier = man.Barrier(num_workers)
        args = [(self._cluster_spec, x, backup_dir, create_barrier) for x in range(num_workers)]
        pool.starmap(testWorkerTrainingState, args)
    except Exception as e:
      print(e, flush=True)
      raise e


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
