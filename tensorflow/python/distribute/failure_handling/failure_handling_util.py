# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Util of GCE specifics to ingegrate with WorkerPreemptionHandler."""
import enum
import os
import sys
import requests

from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging


GCP_METADATA_HEADER = {'Metadata-Flavor': 'Google'}
_GCE_METADATA_URL_ENV_VARIABLE = 'GCE_METADATA_IP'
_RESTARTABLE_EXIT_CODE = 143
GRACE_PERIOD_GCE = 3600


def gce_exit_fn():
  sys.exit(_RESTARTABLE_EXIT_CODE)


def default_tpu_exit_fn():
  """Default exit function to run after saving checkpoint for TPUStrategy.

  For TPUStrategy, we want the coordinator to exit after workers are down so
  that restarted coordinator would not connect to workers scheduled to be
  preempted. This function achieves so by attempting to get a key-value store
  from coordination service, which will block until workers are done and then
  returns with error. Then we have the coordinator sys.exit(42) to re-schedule
  the job.
  """
  logging.info('Waiting for workers to exit...')
  try:
    context.context().get_config_key_value('BLOCK_TILL_EXIT')
  except:  # pylint: disable=bare-except
    logging.info('Restarting cluster due to preemption.')
    sys.exit(42)


def request_compute_metadata(path):
  """Returns GCE VM compute metadata."""
  gce_metadata_endpoint = 'http://' + os.environ.get(
      _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
  req = request.Request(
      '%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, path),
      headers={'Metadata-Flavor': 'Google'})
  info = request.urlopen(req).read()
  if isinstance(info, bytes):
    return info.decode('utf-8')
  else:
    return info


def termination_watcher_function_gce():
  result = request_compute_metadata(
      'instance/maintenance-event') == 'TERMINATE_ON_HOST_MAINTENANCE'
  return result


def on_gcp():
  """Detect whether the current running environment is on GCP."""
  gce_metadata_endpoint = 'http://' + os.environ.get(
      _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')

  try:
    # Timeout in 5 seconds, in case the test environment has connectivity issue.
    # There is not default timeout, which means it might block forever.
    response = requests.get(
        '%s/computeMetadata/v1/%s' %
        (gce_metadata_endpoint, 'instance/hostname'),
        headers=GCP_METADATA_HEADER,
        timeout=5)
    return response.status_code == 200
  except requests.exceptions.RequestException:
    return False


@enum.unique
class PlatformDevice(enum.Enum):
  INTERNAL_CPU = 'internal_CPU'
  INTERNAL_GPU = 'internal_GPU'
  INTERNAL_TPU = 'internal_TPU'
  GCE_GPU = 'GCE_GPU'
  GCE_TPU = 'GCE_TPU'
  GCE_CPU = 'GCE_CPU'
  UNSUPPORTED = 'unsupported'


def detect_platform():
  """Returns the platform and device information."""
  if on_gcp():
    if context.context().list_logical_devices('GPU'):
      return PlatformDevice.GCE_GPU
    elif context.context().list_logical_devices('TPU'):
      return PlatformDevice.GCE_TPU
    else:
      return PlatformDevice.GCE_CPU

  else:
    if context.context().list_logical_devices('GPU'):
      return PlatformDevice.INTERNAL_GPU
    elif context.context().list_logical_devices('TPU'):
      return PlatformDevice.INTERNAL_TPU
    else:
      return PlatformDevice.INTERNAL_CPU
