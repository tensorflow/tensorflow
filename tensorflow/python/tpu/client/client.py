# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# Lint as: python3
"""Cloud TPU Client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import time

from absl import flags
from concurrent import futures
from six.moves.urllib import request
from six.moves.urllib.error import HTTPError

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False

FLAGS = flags.FLAGS

flags.DEFINE_bool('runtime_oom_exit', True,
                  'Exit the script when the TPU runtime is OOM.')
flags.DEFINE_bool('hbm_oom_exit', True,
                  'Exit the script when the TPU HBM is OOM.')

_GKE_ENV_VARIABLE = 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'
_ENDPOINTS_SEPARATOR = ','
_DEFAULT_ENV_VARIABLE = 'TPU_NAME'
_DISCOVERY_SERVICE_URL_ENV_VARIABLE = 'TPU_API_DISCOVERY_URL'
_GCE_METADATA_URL_ENV_VARIABLE = 'GCE_METADATA_IP'
_DEFAULT_ENDPOINT_PORT = '8470'
_OOM_EVENT_COOL_TIME_SEC = 90
_VERSION_SWITCHER_ENDPOINT = 'http://{}:8475/requestversion'


def _utcnow():
  """A wrapper function around datetime.datetime.utcnow.

  This function is created for unit testing purpose. It's not easy to do
  StubOutWithMock with datetime.datetime package.

  Returns:
    datetime.datetime
  """
  return datetime.datetime.utcnow()


def _environment_discovery_url():
  return os.environ.get(_DISCOVERY_SERVICE_URL_ENV_VARIABLE)


def _gce_metadata_endpoint():
  return 'http://' + os.environ.get(_GCE_METADATA_URL_ENV_VARIABLE,
                                    'metadata.google.internal')


def _request_compute_metadata(path):
  req = request.Request(
      '%s/computeMetadata/v1/%s' % (_gce_metadata_endpoint(), path),
      headers={'Metadata-Flavor': 'Google'})
  resp = request.urlopen(req)
  return _as_text(resp.read())


def _environment_var_to_network_endpoints(endpoints):
  """Yields a dict with ip address and port."""
  for endpoint in endpoints.split(','):
    grpc_prefix = 'grpc://'
    if endpoint.startswith(grpc_prefix):
      endpoint = endpoint.split(grpc_prefix)[1]
    parts = endpoint.split(':')
    ip_address = parts[0]
    port = _DEFAULT_ENDPOINT_PORT
    if len(parts) > 1:
      port = parts[1]
    yield {
        'ipAddress': ip_address,
        'port': port
    }


def _get_tpu_name(tpu):
  if tpu:
    return tpu

  for e in [_GKE_ENV_VARIABLE, _DEFAULT_ENV_VARIABLE]:
    if e in os.environ:
      return os.environ[e]
  return None


def _as_text(s):
  if isinstance(s, bytes):
    return s.decode('utf-8')
  return s


class Client(object):
  """Client for working with the Cloud TPU API.

  This client is intended to be used for resolving tpu name to ip addresses.

  It's recommended to use this library as a contextlib to utilize all
  functionality.
  """

  def __init__(self,
               tpu=None,
               zone=None,
               project=None,
               credentials='default',
               service=None,
               discovery_url=None):
    if isinstance(tpu, list):
      if not tpu:
        raise ValueError('At least one TPU must be specified.')
      if len(tpu) != 1:
        raise NotImplementedError(
            'Using multiple TPUs in a single session is not yet implemented')
      tpu = tpu[0]

    tpu = _get_tpu_name(tpu)

    if tpu is None:
      raise ValueError('Please provide a TPU Name to connect to.')

    self._tpu = _as_text(tpu)

    self._use_api = not self._tpu.startswith('grpc://')
    self._service = service

    self._credentials = None
    self._project = None
    self._zone = None
    self._discovery_url = None
    if self._use_api:
      if credentials != 'default':
        self._credentials = credentials
      # Automatically detect project and zone if unspecified.
      if project:
        self._project = _as_text(project)
      else:
        self._project = _request_compute_metadata('project/project-id')
      if zone:
        self._zone = _as_text(zone)
      else:
        zone_path = _request_compute_metadata('instance/zone')
        self._zone = zone_path.split('/')[-1]
      self._discovery_url = _environment_discovery_url() or discovery_url

  def _symptom_msg(self, msg):
    """Return the structured Symptom message."""
    return 'Symptom: ' + msg

  def _oom_event(self, symptoms):
    """Check if a runtime OOM event is reported."""
    if not symptoms:
      return False
    for symptom in reversed(symptoms):
      if symptom['symptomType'] != 'OUT_OF_MEMORY':
        continue
      oom_datetime_str = symptom['createTime'].split('.')[0]
      oom_datetime = datetime.datetime.strptime(oom_datetime_str,
                                                '%Y-%m-%dT%H:%M:%S')
      time_diff = _utcnow() - oom_datetime
      if time_diff < datetime.timedelta(seconds=_OOM_EVENT_COOL_TIME_SEC):
        logging.warning(self._symptom_msg(
            'a recent runtime OOM has occured ~{} seconds ago. The model '
            'script will terminate automatically. To prevent future OOM '
            'events, please consider reducing the model size. To disable this '
            'behavior, set flag --runtime_oom_exit=false when starting the '
            'script.'.format(time_diff.seconds)))
        return True
    return False

  def _hbm_oom_event(self, symptoms):
    """Check if a HBM OOM event is reported."""
    if not symptoms:
      return False
    for symptom in reversed(symptoms):
      if symptom['symptomType'] != 'HBM_OUT_OF_MEMORY':
        continue
      oom_datetime_str = symptom['createTime'].split('.')[0]
      oom_datetime = datetime.datetime.strptime(oom_datetime_str,
                                                '%Y-%m-%dT%H:%M:%S')
      time_diff = _utcnow() - oom_datetime
      if time_diff < datetime.timedelta(seconds=_OOM_EVENT_COOL_TIME_SEC):
        logging.warning(self._symptom_msg(
            'a recent HBM OOM has occured ~{} seconds ago. The model '
            'script will terminate automatically. To prevent future HBM OOM '
            'events, please consider reducing the model size. To disable this '
            'behavior, set flag --hbm_oom_exit=false when starting the '
            'script.'.format(time_diff.seconds)))
        return True
    return False

  def _tpu_service(self):
    """Creates a new Cloud TPU API object.

    This works around an issue where the underlying HTTP connection sometimes
    times out when the script has been running for too long. Other methods in
    this object call this method to get a new API object whenever they need
    to communicate with the Cloud API.

    Raises:
      RuntimeError: If the dependent Python packages are missing.

    Returns:
      A Google Cloud TPU API object.
    """
    if self._service:
      return self._service

    if not _GOOGLE_API_CLIENT_INSTALLED:
      raise RuntimeError('Missing runtime dependency on the Google API client. '
                         'Run `pip install cloud-tpu-client` to fix.')

    credentials = self._credentials
    if credentials is None or credentials == 'default':
      credentials = client.GoogleCredentials.get_application_default()

    if self._discovery_url:
      return discovery.build(
          'tpu',
          'v1',
          credentials=credentials,
          discoveryServiceUrl=self._discovery_url,
          cache_discovery=False)
    else:
      return discovery.build(
          'tpu', 'v1', credentials=credentials, cache_discovery=False)

  def _full_name(self):
    """Returns the full Cloud name for this TPU."""
    return 'projects/%s/locations/%s/nodes/%s' % (
        self._project, self._zone, self._tpu)

  def _fetch_cloud_tpu_metadata(self):
    """Returns the TPU metadata object from the TPU Get API call."""
    service = self._tpu_service()
    try:
      r = service.projects().locations().nodes().get(name=self._full_name())
      return r.execute()
    except Exception as e:
      raise ValueError("Could not lookup TPU metadata from name '%s'. Please "
                       'doublecheck the tpu argument in the TPUClusterResolver '
                       'constructor. Exception: %s' % (self._tpu, e))

  def _get_tpu_property(self, key):
    if self._use_api:
      metadata = self._fetch_cloud_tpu_metadata()
      return metadata.get(key)

    return None

  def __enter__(self):
    self._open = True

  def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
    del type, value, traceback

  def recoverable(self):
    """Returns true if the TPU is in a state where training should eventually resume.

    If false the TPU is in a unrecoverable state and should be recreated.
    """
    state = self.state()
    symptoms = self.symptoms()
    if state and state in ['TERMINATED', 'PREEMPTED']:
      return False
    elif FLAGS.runtime_oom_exit and self._oom_event(symptoms):
      return False
    elif FLAGS.hbm_oom_exit and self._hbm_oom_event(symptoms):
      return False
    return True

  def symptoms(self):
    """Return Cloud TPU Symptoms of the TPU."""
    return self._get_tpu_property('symptoms')

  def state(self):
    """Return state of the TPU."""
    return self._get_tpu_property('state')

  def health(self):
    """Return health of the TPU."""
    return self._get_tpu_property('health')

  def runtime_version(self):
    """Return runtime version of the TPU."""

    if not self._use_api:
      # Fallback on getting version directly from TPU.
      url = _VERSION_SWITCHER_ENDPOINT.format(
          self.network_endpoints()[0]['ipAddress'])
      try:
        req = request.Request(url)
        resp = request.urlopen(req)
        version_details = json.loads(resp.read())
        return version_details.get('currentVersion')
      except HTTPError as e:
        status_code = e.code
        if status_code == 404:
          return None
        else:
          raise e
    return self._get_tpu_property('tensorflowVersion')

  def accelerator_type(self):
    """Return accelerator type of the TPU."""
    return self._get_tpu_property('acceleratorType')

  def api_available(self):
    """Return if the Cloud TPU API is available, if not certain features will not work."""
    return self._use_api

  def name(self):
    """Return the name of the tpu, or the ip address if name is not provided."""
    return self._tpu

  def get_local_ip(self):
    """Return the local ip address of the Google Cloud VM the workload is running on."""
    return _request_compute_metadata('instance/network-interfaces/0/ip')

  def network_endpoints(self):
    """Return a list of tpu endpoints."""
    if not self._use_api:
      return list(_environment_var_to_network_endpoints(self._tpu))
    response = self._fetch_cloud_tpu_metadata()

    if response.get('state') != 'READY':
      raise RuntimeError('TPU "%s" is not yet ready; state: "%s"' %
                         (self._tpu, response.get('state')))
    if 'networkEndpoints' in response:
      return response['networkEndpoints']
    else:
      return [{'ipAddress': response['ipAddress'], 'port': response['port']}]

  def wait_for_healthy(self, timeout_s=1200, interval=30):
    """Wait for TPU to become healthy or raise error if timeout reached.

    Args:
      timeout_s (int): The timeout in seconds for waiting TPU to become healthy.
      interval (int): The interval in seconds to poll the TPU for health.

    Raises:
      RuntimeError: If the TPU doesn't become healthy by the timeout.
    """
    timeout = time.time() + timeout_s
    while self.health() != 'HEALTHY':
      logging.warning(
          ('Waiting for TPU "%s" with state "%s" '
           'and health "%s" to become healthy'),
          self.name(), self.state(), self.health())
      if time.time() + interval > timeout:
        raise RuntimeError(
            'Timed out waiting for TPU "%s" to become healthy' % self.name())
      time.sleep(interval)

    logging.warning('TPU "%s" is healthy.', self.name())

  def configure_tpu_version(self, version, restart_type='always'):
    """Configure TPU software version.

    Args:
      version (string): Version of software to configure the TPU with.
      restart_type (string): Restart behaviour when switching versions,
        defaults to always restart. Options are 'always', 'ifNeeded'.

    """

    def configure_worker(worker):
      """Configure individual TPU worker.

      Args:
        worker: A dict with the field ipAddress where the configure request will
          be sent.
      """
      ip_address = worker['ipAddress']
      url = (_VERSION_SWITCHER_ENDPOINT + '/{}?restartType={}').format(
          ip_address, version, restart_type)
      req = request.Request(url, data=b'')
      try:
        request.urlopen(req)
      except HTTPError as e:
        status_code = e.code
        if status_code == 404:
          raise Exception(
              'Tensorflow version {} is not available on Cloud TPU, '
              'try a previous nightly version or refer to '
              'https://cloud.google.com/tpu/docs/release-notes for '
              'the latest official version.'.format(version))
        else:
          raise Exception('Failed to configure worker {}'.format(ip_address))

    workers = self.network_endpoints()

    with futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
      results = executor.map(configure_worker, workers)
      for result in results:
        if result:
          result.result()
