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
# ===================================================================
"""TPU system metadata and associated tooling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

_PINGING_MASTER_TIMEOUT_IN_MS = 60 * 1000  # 1 min
_RETRY_TIMES = 120
_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS = 300 * 1000  # 5 mins

_TPU_DEVICE_REG = re.compile(r'.*task:(\d+)/.*device:TPU:(\d+)$')

# _TPUSystemMetadata is used by TPUEstimator to hold TPU configuration,
# including num_cores and num_hosts.
_TPUSystemMetadata = collections.namedtuple('_TPUSystemMetadata', [
    'num_cores',
    'num_hosts',
    'num_of_cores_per_host',
    'topology',
    'devices',
])


def _query_tpu_system_metadata(master_address, query_topology=False):
  """Automatically detects the TPU system metadata in the system."""
  tpu_core_count = 0
  devices = []
  device_dict = collections.defaultdict(list)

  retry_count = 1
  while True:
    logging.info('Querying Tensorflow master (%s) for TPU system metadata.',
                 master_address)
    try:
      with ops.Graph().as_default():
        with session_lib.Session(
            master_address,
            config=config_pb2.ConfigProto(
                operation_timeout_in_ms=_PINGING_MASTER_TIMEOUT_IN_MS)) as sess:
          devices = sess.list_devices()
          for device in devices:
            match = _TPU_DEVICE_REG.match(device.name)
            if match:
              host_id = match.group(1)
              core_id = match.group(2)
              device_dict[host_id].append(core_id)
              tpu_core_count += 1
          break
    except errors.DeadlineExceededError:
      msg = ('Fail to connect Tensorflow master. It could be the TPU worker is '
             'not ready (still under scheduling) or Tensorflow '
             'master address is correct: got (%s).' %
             (master_address))

      # TODO(xiejw): For local or grpc master we might not need retry logic
      # here.
      if retry_count <= _RETRY_TIMES:
        logging.warning('%s', msg)
        logging.warning('Retrying (%d/%d).', retry_count, _RETRY_TIMES)
        retry_count += 1
      else:
        raise ValueError(msg)

  num_of_cores_per_host = 0
  if tpu_core_count:
    num_cores_per_host_set = set(
        [len(core_ids) for core_ids in device_dict.values()])
    if len(num_cores_per_host_set) != 1:
      raise RuntimeError(
          'TPU cores on each host is not same. This should not happen!. '
          'devices: {}'.format(devices))
    num_of_cores_per_host = num_cores_per_host_set.pop()

  topology = None
  if query_topology:
    if not tpu_core_count:
      raise RuntimeError(
          'Cannot find any TPU cores in the system (master address {}). '
          'This usually means the master address is incorrect or the '
          'TPU worker has some problems. Available devices: {}'.format(
              master_address, devices))

    topology = _obtain_topology(master_address)

  metadata = _TPUSystemMetadata(
      num_cores=tpu_core_count,
      num_hosts=len(device_dict),
      num_of_cores_per_host=num_of_cores_per_host,
      topology=topology,
      devices=devices)

  msg = 'Found TPU system %s' if tpu_core_count else 'Failed to find TPU: %s'
  logging.info(msg, metadata)
  return metadata


def _obtain_topology(master_address):
  try:
    logging.info('Initializing TPU system (master: %s) to fetch topology '
                 'for model parallelism. This might take a while.',
                 master_address)
    with ops.Graph().as_default():
      session_config = config_pb2.ConfigProto(
          operation_timeout_in_ms=_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS)
      with session_lib.Session(
          master_address, config=session_config) as sess:
        topology = sess.run(tpu.initialize_system())
        return topology
  except errors.DeadlineExceededError:
    raise ValueError(
        'Fail to initialize TPU system with master (%s). '
        'Please double check the TPU system is functional.' % (
            master_address))


