# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Multi-client tests for input_util."""

import os
from typing import Any, List, Mapping, Optional, Tuple

from absl import logging
from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import multi_client_test_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


mp_context = test_backend_util.get_mp_context()

# Multi-client test constants.
JOB_NAME = 'worker'
TF_DATA_SERVICE_JOB_NAME = 'dtensor_tf_data'
NUM_CLIENTS = 4
NUM_DEVICES_PER_CLIENT = 4

# Mesh constants.
MESH_DIM_BATCH = 'batch'
MESH_DIM_HEIGHT = 'height'
MESH_DIM_WIDTH = 'width'

# Data constants.
IMG_HEIGHT = 8
IMG_WIDTH = 8
IMG_CHANNELS = 3

UNSHARDED = layout_lib.UNSHARDED
Mesh = layout_lib.Mesh
Layout = layout_lib.Layout


def redirect_output(file_name):
  # Redirect stderr/stdout to undeclared outputs on sponge.
  artifact_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '')
  if artifact_dir:
    with open(os.path.join(artifact_dir, file_name), 'wb') as fp:
      os.dup2(fp.fileno(), 1)
      os.dup2(fp.fileno(), 2)


def create_dispatcher(test_name, worker_addresses, port, pipe=None):
  dispatcher = server_lib.DispatchServer(
      config=server_lib.DispatcherConfig(
          port=port, protocol='grpc', worker_addresses=worker_addresses
      )
  )
  dispatcher.start()
  if pipe is None:
    # Dispatcher is not within subprocess, so do not block.
    return dispatcher, dispatcher._address
  else:
    redirect_output(f'test-{test_name}-dispatcher.log')
    pipe.send(dispatcher._address)
    signal = pipe.recv()  # blocks until a 'stop' signal is received
    if signal == 'stop':
      dispatcher._stop()
      pipe.send('stopped')
    else:
      raise ValueError('Got unknown signal %s' % signal)


def create_worker(test_name, dispatcher_address, port=None, pipe=None):
  worker = server_lib.WorkerServer(
      config=server_lib.WorkerConfig(
          port=port, dispatcher_address=dispatcher_address, protocol='grpc'
      )
  )
  worker.start()
  if pipe is None:
    # Worker is not within subprocess, so do not block.
    return worker, worker._address
  else:
    redirect_output(f'test-{test_name}-worker.log')
    pipe.send(worker._address)
    signal = pipe.recv()  # blocks until a 'stop' signal is received
    if signal == 'stop':
      worker._stop()
      pipe.send('stopped')
    else:
      raise ValueError('Got unknown signal %s' % signal)


class TFDataServiceCluster:
  """tf.data service cluster with dispatcher and workers as subprocesses.

  To run the cluster in co-located mode, set `num_workers` to 0 and create the
  tf.data service workers manually in each client process.
  """

  def __init__(self,
               test_name,
               num_workers,
               worker_ports=None,
               worker_addresses=None):
    self._test_name = test_name
    self._num_workers = num_workers
    self._start_dispatcher(worker_addresses)
    self._start_workers(worker_ports)

  def _start_dispatcher(self, worker_addresses, port=0):
    self._pipe_to_dispatcher, dispatcher_pipe = mp_context.Pipe(True)
    logging.info(
        'Starting remote dispatcher on port %d with worker addresses: %s', port,
        worker_addresses)
    self._dispatcher_process = mp_context.Process(
        target=create_dispatcher,
        args=(self._test_name, worker_addresses, port, dispatcher_pipe),
    )
    self._dispatcher_process.start()
    self._dispatcher_address = self._pipe_to_dispatcher.recv()

  def dispatcher_address(self):
    return self._dispatcher_address

  def _start_workers(self, worker_ports=None):
    self._workers = []
    self._worker_addresses = []
    self._worker_pipes = []
    for idx in range(self._num_workers):
      port = worker_ports[idx] if worker_ports else None
      self._start_worker(port)

  def _start_worker(self, port=None):
    pipe_to_worker, worker_pipe = mp_context.Pipe(True)
    logging.info(
        'Starting remote worker on port %d with dispatcher address: %s', port,
        self._dispatcher_address)
    worker_process = mp_context.Process(
        target=create_worker,
        args=(self._test_name, self._dispatcher_address, port, worker_pipe),
    )
    worker_process.start()
    worker_address = self._pipe_to_worker.recv()
    self._workers.append(worker_process)
    self._worker_addresses.append(worker_address)
    self._worker_pipes.append(pipe_to_worker)

  def worker_addresses(self):
    return self._worker_addresses

  def stop(self):
    # Segfault logs may still be printed because clean exit of child processes
    # is not always possible. This will not affect the outcome of the test.
    logging.info('Will try to stop TFDataServiceCluster!')

    for idx in range(self._num_workers):
      address = self._worker_addresses[idx]
      pipe_to_worker = self._worker_pipes[idx]
      logging.info('Stopping worker %s...', address)
      pipe_to_worker.send('stop')
      if pipe_to_worker.poll(2):
        if pipe_to_worker.recv() == 'stopped':
          logging.info('Successfully stopped worker %s', address)
      self._workers[idx].terminate()

    logging.info('Stopping dispatcher...')
    self._pipe_to_dispatcher.send('stop')
    if self._pipe_to_dispatcher.poll(2):
      if self._pipe_to_dispatcher.recv() == 'stopped':
        logging.info('Successfully stopped dispatcher')
    self._dispatcher_process.terminate()


def setup_local_devices(num_devices):
  physical_cpus = tf_config.list_physical_devices('CPU')
  tf_config.set_logical_device_configuration(
      physical_cpus[0],
      [context.LogicalDeviceConfiguration() for _ in range(num_devices)],
  )


def setup_client(client_id: int, test_name: str, env: Mapping[str, str],
                 num_local_devices: int):
  """Set up a DTensor client for use in multi-client tests.

  Args:
    client_id: the index of the client.
    test_name: the name of the test under which this client is running, used To
      identify the log file artifact containing the test output.
    env: a dictionary of environment variables to update.
    num_local_devices: number of local devices to set up.
  """
  # Redirect client's stderr/stdout to undeclared outputs on sponge.
  redirect_output(f'test-{test_name}-process-{client_id}.log')

  # Update any specified environment variables.
  for var, val in env.items():
    os.environ[var] = val

  # Set up local devices.
  setup_local_devices(num_local_devices)

  # Set up DTensor cluster and enable collectives.
  accelerator_util.initialize_accelerator_system()


def run_client(
    client_id: int,
    test_name: str,
    env: Mapping[str, str],
    num_local_devices: int,
    dispatcher_address: str,
    worker_port: int,
    batch_size: int,
    dataset_paths: List[str],
    mesh: Mesh,
    batch_dim: Optional[str],
    layouts: Tuple[Layout, Layout],
) -> List[Tuple[Any, Any]]:
  # Co-located tf.data service mode. It is important to hold the worker object
  # until the end otherwise it will get garbage collected.
  worker, worker_address = create_worker(  # pylint: disable=unused-variable
      test_name, dispatcher_address, port=worker_port)
  logging.info(
      'tf.data service worker running at %s',
      worker_address,
  )

  setup_client(client_id, test_name, env, num_local_devices)

  def decode_fn(record_bytes):
    decoded = parsing_ops.parse_single_example_v2(
        serialized=record_bytes,
        features={
            'idx': parsing_config.FixedLenFeature([], dtype=dtypes.int64),
            'elem': parsing_config.FixedLenFeature([], dtype=dtypes.string),
        },
    )
    parsed_elem = gen_parsing_ops.parse_tensor(decoded['elem'], dtypes.int32)
    elem = check_ops.ensure_shape(
        parsed_elem, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
    )
    return decoded['idx'], elem

  dataset = dataset_ops.DatasetV2.from_tensor_slices(dataset_paths)
  dataset = dataset.interleave(readers.TFRecordDatasetV2)
  dataset = dataset.map(decode_fn)

  tf_data_service_config = input_util.TFDataServiceConfig(
      dispatcher_address=dispatcher_address, job_name=TF_DATA_SERVICE_JOB_NAME
  )
  d_dataset = input_util.DTensorDataset(
      dataset=dataset,
      global_batch_size=batch_size,
      mesh=mesh,
      layouts=layouts,
      batch_dim=batch_dim,
      tf_data_service_config=tf_data_service_config,
  )

  # Subprocesses cannot return a sharded DTensor as it triggers a copy and
  # copying non-replicated DTensors is not supported. So instead we unpack it
  # and return the component tensors.
  ret = []
  for batch_idx, elem in d_dataset:
    n_batch_idx = api.unpack(batch_idx)
    n_elem = api.unpack(elem)
    ret.append((n_batch_idx, n_elem))
  return ret


class MultiClientDTensorDatasetTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()

    logging.info('Check per client log in Test artifacts.')

    self.server_ports = [
        multi_client_test_util.pick_unused_port() for _ in range(NUM_CLIENTS)
    ]

    self.worker_ports = [
        multi_client_test_util.pick_unused_port() for _ in range(NUM_CLIENTS)
    ]
    worker_addresses = [f'localhost:{port}' for port in self.worker_ports]
    self.cluster = TFDataServiceCluster(
        test_name=self._testMethodName,
        num_workers=0,  # Co-located mode.
        worker_addresses=worker_addresses)

  def tearDown(self):
    super().tearDown()
    self.cluster.stop()

  def write_dataset(self, dataset, num_files, num_elems):
    """Writes a dataset_ops.DatasetV2 to multiple files."""
    dataset_paths = []
    dataset_iter = iter(dataset)

    for file_idx in range(num_files):
      dataset_path = os.path.join(self.get_temp_dir(),
                                  f'dataset-{file_idx}.tfrecords')
      dataset_paths.append(dataset_path)
      with tf_record.TFRecordWriter(dataset_path) as writer:
        for _ in range(num_elems // num_files):
          idx, elem = next(dataset_iter)
          elem_bytes = example_pb2.Example(
              features=feature_pb2.Features(
                  feature={
                      'idx': feature_pb2.Feature(
                          int64_list=feature_pb2.Int64List(value=[idx])
                      ),
                      'elem': feature_pb2.Feature(
                          bytes_list=feature_pb2.BytesList(
                              value=[io_ops.serialize_tensor(elem).numpy()]
                          )
                      ),
                  }
              )
          ).SerializeToString()
          writer.write(elem_bytes)

    return dataset_paths

  @parameterized.product(
      (
          {
              # batch=4 x height=2 x width=2
              # 1 replica per client.
              'mesh_dims': [(MESH_DIM_BATCH, 4),
                            (MESH_DIM_HEIGHT, 2),
                            (MESH_DIM_WIDTH, 2)],
          }, {
              # batch=4 x height=2 x width=2 (transposed)
              # 1 replica per client with reordered local partitions.
              'mesh_dims': [(MESH_DIM_BATCH, 4),
                            (MESH_DIM_WIDTH, 2),
                            (MESH_DIM_HEIGHT, 2)],
          }, {
              # batch=8 x height=2 x width=1
              # 2 replicas per client.
              'mesh_dims': [(MESH_DIM_BATCH, 8),
                            (MESH_DIM_HEIGHT, 2),
                            (MESH_DIM_WIDTH, 1)],
          }, {
              # batch=8 x height=2 x width=1 (transposed)
              # 2 replicas per client with reordered partitions.
              'mesh_dims': [(MESH_DIM_BATCH, 8),
                            (MESH_DIM_WIDTH, 1),
                            (MESH_DIM_HEIGHT, 2)],
          }, {
              # batch=2 x height=4 x width=2
              # 1 replica split over 2 clients.
              'mesh_dims': [(MESH_DIM_BATCH, 2),
                            (MESH_DIM_HEIGHT, 4),
                            (MESH_DIM_WIDTH, 2)],
          }, {
              # batch=2 x height=4 x width=2 (transposed)
              # 1 replica split over 2 clients with reordered partitions.
              'mesh_dims': [(MESH_DIM_BATCH, 2),
                            (MESH_DIM_WIDTH, 2),
                            (MESH_DIM_HEIGHT, 4)],
          },
      ),
      (
          {
              # Replicated
              'idx_sharding': [UNSHARDED],
              'images_sharding': [UNSHARDED, UNSHARDED, UNSHARDED, UNSHARDED],
          }, {
              # Batch sharded
              'idx_sharding': [MESH_DIM_BATCH],
              'images_sharding':
                  [MESH_DIM_BATCH, UNSHARDED, UNSHARDED, UNSHARDED],
          }, {
              # Spatially sharded
              'idx_sharding': [UNSHARDED],
              'images_sharding':
                  [UNSHARDED, MESH_DIM_HEIGHT, MESH_DIM_WIDTH, UNSHARDED],
          }, {
              # Batch and spatially sharded
              'idx_sharding': [MESH_DIM_BATCH],
              'images_sharding':
                  [MESH_DIM_BATCH, MESH_DIM_HEIGHT, MESH_DIM_WIDTH, UNSHARDED],
          }
      ))
  def testMultiClientIter(self, mesh_dims, idx_sharding, images_sharding):
    num_batches = 4
    batch_size = 16
    num_elems = num_batches * batch_size

    images = stateless_random_ops.stateless_random_uniform(
        [num_elems, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS],
        seed=(1, 2),
        minval=0,
        maxval=255,
        dtype=dtypes.int32,
    )
    dataset = dataset_ops.DatasetV2.from_tensor_slices(images)

    # Enumerate the dataset elements to make it easier to identify the batches
    # returned by the DTensorDataset.
    dataset = dataset.enumerate()

    # Store a mapping of index to dataset elements which can be looked up later
    # to identify the batches returned by the DTensorDataset.
    all_elems = {idx.numpy(): elem for idx, elem in dataset}

    # Write the dataset and shard it among multiple files.
    dataset_paths = self.write_dataset(
        dataset, num_files=8, num_elems=num_elems)

    # Construct args for starmap.
    args = []
    mesh_dim_names, mesh_dim_sizes = zip(*mesh_dims)
    global_device_ids = test_util.create_device_ids_array(mesh_dim_sizes)
    device_ids_split = np.split(np.ravel(global_device_ids), NUM_CLIENTS)
    dtensor_jobs = [
        f'localhost:{self.server_ports[i]}' for i in range(NUM_CLIENTS)
    ]

    for client_id in range(NUM_CLIENTS):
      # Manually specify DTensor environment variables since we are in a test
      # environment.
      env = {
          config._DT_CLIENT_ID: str(client_id),
          config._DT_JOB_NAME: str(JOB_NAME),
          config._DT_JOBS: ','.join(dtensor_jobs)
      }

      local_device_ids = device_ids_split[client_id].tolist()
      local_devices = [
          device_spec.DeviceSpecV2(  # pylint: disable=g-complex-comprehension
              job=JOB_NAME,
              replica=0,
              task=client_id,
              device_type='CPU',
              device_index=i,
          )
          for i in range(len(local_device_ids))
      ]
      mesh = Mesh(
          dim_names=mesh_dim_names,
          global_device_ids=global_device_ids,
          local_device_ids=local_device_ids,
          local_devices=local_devices,
      )
      idx_layout = Layout(idx_sharding, mesh)
      images_layout = Layout(images_sharding, mesh)
      batch_dim = MESH_DIM_BATCH if MESH_DIM_BATCH in images_sharding else None

      args.append((client_id, self._testMethodName, env, NUM_DEVICES_PER_CLIENT,
                   self.cluster.dispatcher_address(),
                   self.worker_ports[client_id], batch_size, dataset_paths,
                   mesh, batch_dim, (idx_layout, images_layout)))

    def get_results():
      # Run the DTensor client processes and get the DTensor dataset components.
      with mp_context.Pool(NUM_CLIENTS) as pool:
        results = pool.starmap(run_client, args)
        pool.close()
        pool.join()

      return results

    # TODO(b/271162918): fix multi-client use case.
    with self.assertRaises(NotImplementedError):
      results = get_results()

    return
    # pylint: disable=unreachable

    # Create a mesh on the main test process. The tensor components returned
    # from each DTensor client subprocess will be packed onto this mesh to
    # verify correctness.
    test_mesh = mesh_util.create_mesh(
        mesh_dims=mesh_dims,
        devices=[
            'CPU:%d' % i for i in range(NUM_CLIENTS * NUM_DEVICES_PER_CLIENT)
        ])
    test_mesh = self.configTestMesh({'CPU': test_mesh})
    idx_test_layout = Layout(idx_sharding, test_mesh)
    images_test_layout = Layout(images_sharding, test_mesh)

    for batch_elems in zip(*results):
      # Collect the tensor components returned from each client.
      idx_components = []
      images_components = []
      for client_id in range(NUM_CLIENTS):
        local_idx, local_images = batch_elems[client_id]
        idx_components.extend(local_idx)
        images_components.extend(local_images)

      # Pack the dataset elements into a DTensor on the test mesh.
      d_idx = api.pack(idx_components, idx_test_layout)
      d_images = api.pack(images_components, images_test_layout)

      # Get the batch of elements from the original dataset using the element
      # indices.
      batch_stack = []
      for elem_idx in d_idx:
        batch_stack.append(all_elems.pop(elem_idx.numpy()))
      batch = array_ops_stack.stack(batch_stack)

      self.assertDTensorEqual(batch, images_test_layout, d_images)

    self.assertEmpty(
        all_elems, 'Not all batches were returned by DTensorDataset.')


if __name__ == '__main__':
  test_backend_util.handle_test_main(test.main)
