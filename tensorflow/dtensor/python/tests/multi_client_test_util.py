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
"""Utilities for multi-client setup."""
import os
import sys

from absl import flags
import portpicker

from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.python.platform import test as tf_test


_NUM_LOCAL_DEVICES = flags.DEFINE_integer(
    'num_local_devices', 4,
    'Number of local devices. 4 is the only allowed value for TPU.')
_NUM_CLIENTS = flags.DEFINE_integer(
    'num_clients', 2,
    'Number of clients. 0 for local mode. 2 is the only allowed value for TPU.')


def pick_unused_port():
  """Helper function to return an unused port."""
  return portpicker.pick_unused_port()


def multi_client_main(client_config_function):
  """Creates a Flock of TensorFlow Processes on localhost."""
  flags.FLAGS(sys.argv, known_only=True)
  num_clients = _NUM_CLIENTS.value
  num_process = num_clients or 1
  num_local_devices = _NUM_LOCAL_DEVICES.value

  # No GPU visible to the flock controller.
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.environ['HIP_VISIBLE_DEVICES'] = ''

  # Python multiprocess module in OSS.
  mp_context = test_backend_util.get_mp_context()

  print('Check per client log in Test artifacts.', flush=True)

  # Inverts the order of ports intentionally to rule out ordering bugs.
  server_ports = sorted(
      [pick_unused_port() for _ in range(num_process)], reverse=True
  )

  additional_ports = sorted([pick_unused_port() for _ in range(num_process)])

  # Starts processes
  procs = []
  for client_idx in range(num_process):
    proc = mp_context.Process(
        target=run_client,
        args=(client_idx, num_clients, server_ports, additional_ports,
              num_local_devices, client_config_function),
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


def run_client(idx, num_clients, server_ports, additional_ports,
               num_local_devices, client_config_function):
  """Runs test.main() from a DTensor Client process on localhost.

  This function runs in a separate process so that the eager context is
  properly separated, which resembles real world multi-client setup.

  Virtual devices are configured before test.main() is called.

  Each client is configured to only have access to the physical GPU device
  corresponding to its client id via CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES.

  Each client is configured to only have access to some TPU cores
  corresponding to its client id via flags.

  The clients redirect stdout and stderr to files under Test Artifacts.

  Args:
    idx: integer task number represents the client's id from global picture.
    num_clients: total number of clients.
    server_ports: A list of ports that is allocated and to be used to construct
      GRPC server. server_ports[idx] will be the GRPC server on the
      corresponding client.
    additional_ports: A list of ports that is allocated and to be used to
      construct the backends.
    num_local_devices: Number of devices per client.
    client_config_function: A function, for each of the client to config the
      local environment variables, etc. Note that the function will be called
      with a dict of extra params, eg:
        {'num_clients': 2
         'client_id': 0,
         'worker_jobs': ['localhost:port1', 'localhost:port2'],
         'num_devices': 4,
        }
  """
  test_backend_util.slice_host_devices_for_multiworker(
      num_clients, idx, additional_ports
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
  worker_jobs = [f'localhost:{port:06d}' for port in server_ports]
  client_config_func_param = {
      'num_clients': num_clients,
      'client_id': idx,
      'worker_jobs': worker_jobs,
      'num_devices': num_local_devices,
  }
  client_config_function(client_config_func_param)

  # The following function call never returns.
  tf_test.main()
