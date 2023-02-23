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
"""Tests for DTensor multi-client setup."""
import os
import sys

from absl import flags
import numpy as np
import portpicker

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as d_layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test


_NUM_DEVICES = flags.DEFINE_integer(
    'num_devices', 4, 'Number of local devices. '
    '4 is the only allowed value for TPU.')
_NUM_CLIENTS = flags.DEFINE_integer(
    'num_clients', 2, 'Number of clients. 0 for local mode.'
    '2 is the only allowed value for TPU.')
_MODEL_DIM_SIZE = flags.DEFINE_integer('model_dim_size', 4,
                                       'Size of the model dimension.')

_BATCH_DIM = 'batch'
_MODEL_DIM = 'model'

_BATCH_SIZE = 8
_STEPS = 5
_LR = 1e-3


@polymorphic_function.function
def _run_step(inputs, w, b, k):
  with backprop.GradientTape() as g:
    g.watch([w, b])
    logits = nn_ops.conv2d_v2(inputs, k, strides=[1, 1, 1, 1], padding='SAME')
    logits = array_ops.reshape(logits, [logits.shape[0], -1])
    logits = math_ops.matmul(logits, w)
    logits = logits + b
    loss = math_ops.reduce_sum(logits, axis=[0, 1])

  gw, gb = g.gradient(loss, [w, b])
  for v, v_grad in zip([w, b], [gw, gb]):
    v.assign_sub(_LR * v_grad)
  return gw, gb, loss


def init_var(mesh):
  w_initializer = stateless_random_ops.stateless_random_normal(
      [28 * 28, 16], seed=[0, 1]
  )
  b_initializer = stateless_random_ops.stateless_random_normal(
      [16], seed=[0, 2]
  )
  k_initializer = stateless_random_ops.stateless_random_normal(
      [3, 3, 1, 1], seed=[0, 3]
  )
  n_w = variables.Variable(w_initializer)
  n_b = variables.Variable(b_initializer)
  n_k = variables.Variable(k_initializer)

  # Initialize DTensor variables.
  w_initializer_on_mesh = d_api.copy_to_mesh(
      w_initializer, d_layout.Layout.replicated(mesh, rank=2)
  )
  b_initializer_on_mesh = d_api.copy_to_mesh(
      b_initializer, d_layout.Layout.replicated(mesh, rank=1)
  )
  k_initializer_on_mesh = d_api.copy_to_mesh(
      k_initializer, d_layout.Layout.replicated(mesh, rank=4)
  )
  w = d_variable.DVariable(
      d_api.relayout(
          w_initializer_on_mesh,
          d_layout.Layout(['unsharded', _MODEL_DIM], mesh),
      )
  )
  b = d_variable.DVariable(
      d_api.relayout(b_initializer_on_mesh, d_layout.Layout([_MODEL_DIM], mesh))
  )
  k = d_variable.DVariable(k_initializer_on_mesh)
  return (n_w, n_b, n_k), (w, b, k)


class DTensorMNISTTest(tf_test.TestCase):

  def setUp(self):
    super(DTensorMNISTTest, self).setUp()
    if config.list_physical_devices('GPU'):
      device_type = 'GPU'
    elif test_util.is_tpu_present():
      device_type = 'TPU'
    else:
      device_type = 'CPU'

    local_devices = d_config.local_devices(device_type)
    num_devices = len(local_devices)

    global_device_ids = test_util.create_device_ids_array((
        d_config.num_clients() * num_devices // _MODEL_DIM_SIZE.value,
        _MODEL_DIM_SIZE.value,
    ))
    device_ids_list = np.ravel(global_device_ids).tolist()

    index = d_config.client_id() * num_devices
    local_device_ids = device_ids_list[index:(index + num_devices)]

    self.mesh = d_layout.Mesh(
        [_BATCH_DIM, _MODEL_DIM],
        global_device_ids=global_device_ids,
        local_device_ids=local_device_ids,
        local_devices=local_devices,
    )

  def tearDown(self):
    # A barrier prevents a client from disconnecting prematurely in tests.
    mesh_util.barrier(self.mesh)
    super().tearDown()

  def test_mnist(self):
    def train():
      input_layout = d_layout.Layout.batch_sharded(
          self.mesh, _BATCH_DIM, rank=4
      )
      (n_w, n_b, n_k), (w, b, k) = init_var(self.mesh)
      for i in range(_STEPS):
        data = stateless_random_ops.stateless_random_normal(
            [_BATCH_SIZE, 28, 28, 1], seed=[0, i]
        )
        g_nw, g_nb, n_loss = _run_step(data.numpy(), n_w, n_b, n_k)

        input_image = d_api.copy_to_mesh(
            data, layout=d_layout.Layout.replicated(self.mesh, rank=4)
        )
        input_image = d_api.relayout(input_image, layout=input_layout)

        with ops.device_v2(self.mesh.local_devices()[0]):
          gw, gb, loss = _run_step(input_image, w, b, k)

      gw = d_api.relayout(gw, d_layout.Layout.replicated(self.mesh, rank=2))
      w = d_api.relayout(w, d_layout.Layout.replicated(self.mesh, rank=2))
      gb = d_api.relayout(gb, d_layout.Layout.replicated(self.mesh, rank=1))
      b = d_api.relayout(b, d_layout.Layout.replicated(self.mesh, rank=1))

      return (n_loss, g_nw, g_nb, n_w, n_b), (loss, gw, gb, w, b)

    (n_loss, g_nw, g_nb, n_w, n_b), (loss, gw, gb, w, b) = train()

    self.assertAllClose(n_loss, loss, atol=5e-4)
    self.assertAllClose(g_nw, gw, atol=1e-5)
    self.assertAllClose(g_nb, gb, atol=1e-5)
    self.assertAllClose(n_w, w, atol=1e-5)
    self.assertAllClose(n_b, b, atol=1e-5)

  def test_copy_to_mesh(self):
    layout = d_layout.Layout([_BATCH_DIM, 'unsharded'], self.mesh)
    host_layout = d_layout.Layout(layout.sharding_specs, self.mesh.host_mesh())
    x = d_api.pack(
        [array_ops.ones((2, 2), dtype=dtypes.float32)]
        * len(self.mesh.local_devices()),
        layout,
    )

    @polymorphic_function.function
    def d2h(x):
      return d_api.copy_to_mesh(x, host_layout)

    @polymorphic_function.function
    def h2d(x):
      return d_api.copy_to_mesh(x, layout)

    y = d2h(x)
    ys = d_api.unpack(y)
    for i in ys:
      self.assertAllClose(i, array_ops.ones((2, 2)), atol=1e-5)
    z = h2d(y)
    zs = d_api.unpack(z)
    for i in zs:
      self.assertAllClose(i, array_ops.ones((2, 2)), atol=1e-5)


def multi_client_main():
  """Creates a Flock of TensorFlow Processes on localhost."""
  flags.FLAGS(sys.argv, known_only=True)
  num_clients = _NUM_CLIENTS.value or 1
  num_devices = _NUM_DEVICES.value

  # No GPU visible to the flock controller.
  os.environ['CUDA_VISIBLE_DEVICES'] = ''

  # Python multiprocess module in OSS.
  mp_context = test_backend_util.get_mp_context()

  print('Check per client log in Test artifacts.', flush=True)

  # Inverts the order of ports intentionally to rule out ordering bugs.
  server_ports = sorted(
      [portpicker.pick_unused_port() for _ in range(num_clients)], reverse=True)

  additional_ports = sorted(
      [portpicker.pick_unused_port() for _ in range(num_clients)]
  )

  # Starts processes
  procs = []
  for client_idx in range(num_clients):
    proc = mp_context.Process(
        target=run_client,
        args=(client_idx, server_ports, additional_ports, num_devices),
        name=f'Client-{client_idx}',
    )
    proc.start()
    procs.append(proc)

  # Joins processes
  exitcode = 0
  for proc in procs:
    proc.join()
    if proc.exitcode != 0:
      exitcode = proc.exitcode

  sys.exit(exitcode)


def run_client(idx, server_ports, additional_ports, num_devices):
  """Runs test.main() from a DTensor Client process on localhost.

  This function runs in a separate process so that the eager context is
  proprely separated, which resembles real world multi-client setup.

  Virtual devices are configured before test.main() is called.

  Each client is configured to only have access to the physical GPU device
  corresponding to its client id via CUDA_VISIBLE_DEVICES.

  Each client is configured to only have access to some TPU cores
  corresponding to its client id via flags.

  The clients redirects stdout and stderr to files under Test Artifacts.

  Args:
    idx: integer task number represents the client's id from global picture.
    server_ports: A list of ports that is allocated and to be used to construct
      GRPC server. server_ports[idx] will be the GRPC server on the
      corresponding client.
    additional_ports: A list of ports that is allocated and to be used to
      construct the backends.
    num_devices: Number of devices per client.
  """
  # Python ForkServer doesn't parse the absl flags.
  flags.FLAGS(sys.argv, known_only=True)

  test_backend_util.slice_host_devices_for_multiworker(
      _NUM_CLIENTS.value, idx, additional_ports
  )

  artifact_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '')

  # Redirect extra client's stderr/stdout to undeclared outputs on sponge.
  if artifact_dir:
    with open(
        os.path.join(artifact_dir, f'test-client-process-{idx}.log'),
        'wb') as fp:
      os.dup2(fp.fileno(), 1)
      os.dup2(fp.fileno(), 2)

  # Set up cluster and enable collectives.
  dtensor_jobs = [f'localhost:{port:06d}' for port in server_ports]

  # Configures DTensor multi-client environment variables.
  # pylint: disable=protected-access
  if _NUM_CLIENTS.value != 0:
    os.environ[d_config._DT_CLIENT_ID] = f'{idx}'
    os.environ[d_config._DT_JOB_NAME] = 'worker'
    os.environ[d_config._DT_JOBS] = ','.join(dtensor_jobs)
  # pylint: enable=protected-access

  if config.list_physical_devices('GPU'):
    device_type = 'GPU'
  elif test_util.is_tpu_present():
    device_type = 'TPU'
  else:
    device_type = 'CPU'

  reset_logical_devices(device_type, num_devices)

  # The following function call never returns.
  tf_test.main()


def reset_logical_devices(device_type, num_devices):
  """Ensures multi-client with the number of logical devices for CPU/GPU/TPU."""
  test_util.reset_context()
  if device_type != 'TPU':
    # Configure virtual devices. This does not initialize the TensorFlow
    # context.
    test_util.reset_logical_devices(device_type, num_devices)

  accelerator_util.initialize_accelerator_system(
      device_type, enable_coordination_service=True)

  # Validates the correct number of devices are created.
  logical_devices = test_util.list_local_logical_devices(device_type)
  assert len(logical_devices) == num_devices, (
      logical_devices,
      f'Test is misconfigured: expecting {num_devices} logical_devices.')


if __name__ == '__main__':
  test_backend_util.handle_test_main(multi_client_main)
