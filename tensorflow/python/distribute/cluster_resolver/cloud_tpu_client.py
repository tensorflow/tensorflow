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

import os

from six.moves.urllib import request

from tensorflow.python.util import compat

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from apiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False

_GKE_ENV_VARIABLE = 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'
_ENDPOINTS_SEPARATOR = ','
_DEFAULT_ENV_VARIABLE = 'TPU_NAME'
_DISCOVERY_SERVICE_URL_ENV_VARIABLE = 'TPU_API_DISCOVERY_URL'
_GCE_METADATA_ENDPOINT = 'http://metadata.google.internal'
_DEFAULT_ENDPOINT_PORT = '8470'


def _environment_discovery_url():
  return os.environ.get(_DISCOVERY_SERVICE_URL_ENV_VARIABLE)


def _request_compute_metadata(path):
  req = request.Request(
      '%s/computeMetadata/v1/%s' % (_GCE_METADATA_ENDPOINT, path),
      headers={'Metadata-Flavor': 'Google'})
  resp = request.urlopen(req)
  return compat.as_bytes(resp.read())


def _environment_var_to_network_endpoints(endpoints):
  """Yields a dict with ip address and port."""
  for endpoint in endpoints.split(compat.as_text(',')):
    grpc_prefix = compat.as_text('grpc://')
    if endpoint.startswith(grpc_prefix):
      endpoint = endpoint.split(grpc_prefix)[1]
    parts = endpoint.split(compat.as_text(':'))
    ip_address = parts[0]
    port = _DEFAULT_ENDPOINT_PORT
    if len(parts) > 1:
      port = parts[1]
    yield {
        'ipAddress': compat.as_text(ip_address),
        'port': compat.as_text(port)
    }


def _get_tpu_name(tpu):
  if tpu:
    return tpu

  for e in [_GKE_ENV_VARIABLE, _DEFAULT_ENV_VARIABLE]:
    if e in os.environ:
      return os.environ[e]
  return None


class CloudTPUClient(object):
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

    self._tpu = compat.as_text(tpu)

    self._use_api = not tpu.startswith('grpc://')
    self._service = service

    self._credentials = None
    self._project = None
    self._zone = None
    self._discovery_url = None
    if self._use_api:
      if credentials != 'default':
        self._credentials = credentials
      # Automaically detect project and zone if unspecified.
      if project:
        self._project = project
      else:
        self._project = compat.as_str(
            _request_compute_metadata('project/project-id'))
      if zone:
        self._zone = zone
      else:
        zone_path = compat.as_str(_request_compute_metadata('instance/zone'))
        self._zone = zone_path.split('/')[-1]
      self._discovery_url = _environment_discovery_url() or discovery_url

  def _tpu_service(self):
    """Creates a new Cloud TPU API object.

    This works around an issue where the underlying HTTP connection sometimes
    times out when the script has been running for too long. Other methods in
    this object call this method to get a new API object whenever they need
    to communicate with the Cloud API.

    Returns:
      A Google Cloud TPU API object.
    """
    if self._service:
      return self._service

    credentials = self._credentials
    if credentials is None or credentials == 'default':
      credentials = GoogleCredentials.get_application_default()

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

  def _fetch_cloud_tpu_metadata(self):
    """Returns the TPU metadata object from the TPU Get API call."""
    try:
      full_name = 'projects/%s/locations/%s/nodes/%s' % (
          self._project, self._zone, compat.as_text(self._tpu))
      service = self._tpu_service()
      r = service.projects().locations().nodes().get(name=full_name)
      return r.execute()
    except Exception as e:
      raise ValueError("Could not lookup TPU metadata from name '%s'. Please "
                       'doublecheck the tpu argument in the TPUClusterResolver '
                       'constructor. Exception: %s' % (self._tpu, e))

  def __enter__(self):
    self._open = True

  def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
    del type, value, traceback

  def recoverable(self):
    """Returns true if the TPU is in a state where training should eventually resume.

    If false the TPU is in a unrecoverable state and should be recreated.
    """
    state = self.state()
    if state and state in ['TERMINATED', 'PREEMPTED']:
      return False
    return True

  def state(self):
    """Return state of the TPU."""
    if self._use_api:
      metadata = self._fetch_cloud_tpu_metadata()
      if 'state' in metadata:
        return metadata['state']

    return None

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
    response = self._fetch_cloud_tpu_metadata()  # pylint: disable=protected-access

    if 'state' in response and response['state'] != 'READY':
      raise RuntimeError('TPU "%s" is not yet ready; state: "%s"' %
                         (compat.as_text(self._tpu), response['state']))
    if 'networkEndpoints' in response:
      return response['networkEndpoints']
    else:
      return [{'ipAddress': response['ipAddress'], 'port': response['port']}]
