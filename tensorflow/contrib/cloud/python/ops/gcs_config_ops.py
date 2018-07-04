# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""GCS file system configuration for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorflow.contrib.cloud.python.ops import gen_gcs_config_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import training


# @tf_export('contrib.cloud.BlockCacheParams')
class BlockCacheParams(object):
  """BlockCacheParams is a struct used for configuring the GCS Block Cache."""

  def __init__(self, block_size=None, max_bytes=None, max_staleness=None):
    self._block_size = block_size or 128 * 1024 * 1024
    self._max_bytes = max_bytes or 2 * self._block_size
    self._max_staleness = max_staleness or 0

  @property
  def block_size(self):
    return self._block_size

  @property
  def max_bytes(self):
    return self._max_bytes

  @property
  def max_staleness(self):
    return self._max_staleness


# @tf_export('contrib.cloud.ConfigureGcsHook')
class ConfigureGcsHook(training.SessionRunHook):
  """ConfigureGcsHook configures GCS when used with Estimator/TPUEstimator.

  Warning: GCS `credentials` may be transmitted over the network unencrypted.
  Please ensure that the network is trusted before using this function. For
  users running code entirely within Google Cloud, your data is protected by
  encryption in between data centers. For more information, please take a look
  at https://cloud.google.com/security/encryption-in-transit/.

  Example:

  ```
  sess = tf.Session()
  refresh_token = raw_input("Refresh token: ")
  client_secret = raw_input("Client secret: ")
  client_id = "<REDACTED>"
  creds = {
      "client_id": client_id,
      "refresh_token": refresh_token,
      "client_secret": client_secret,
      "type": "authorized_user",
  }
  tf.contrib.cloud.configure_gcs(sess, credentials=creds)
  ```

  """

  def _verify_dictionary(self, creds_dict):
    if 'refresh_token' in creds_dict or 'private_key' in creds_dict:
      return True
    return False

  def __init__(self, credentials=None, block_cache=None):
    """Constructs a ConfigureGcsHook.

    Args:
      credentials: A json-formatted string.
      block_cache: A `BlockCacheParams`

    Raises:
      ValueError: If credentials is improperly formatted or block_cache is not a
        BlockCacheParams.
    """
    if credentials is not None:
      if isinstance(credentials, str):
        try:
          data = json.loads(credentials)
        except ValueError as e:
          raise ValueError('credentials was not a well formed JSON string.', e)
        if not self._verify_dictionary(data):
          raise ValueError(
              'credentials has neither a "refresh_token" nor a "private_key" '
              'field.')
      elif isinstance(credentials, dict):
        if not self._verify_dictionary(credentials):
          raise ValueError('credentials has neither a "refresh_token" nor a '
                           '"private_key" field.')
        credentials = json.dumps(credentials)
      else:
        raise ValueError('credentials is of an unknown type')

    self._credentials = credentials

    if block_cache and not isinstance(block_cache, BlockCacheParams):
      raise ValueError('block_cache must be an instance of BlockCacheParams.')
    self._block_cache = block_cache

  def begin(self):
    if self._credentials:
      self._credentials_placeholder = array_ops.placeholder(dtypes.string)
      self._credentials_op = gen_gcs_config_ops.gcs_configure_credentials(
          self._credentials_placeholder)
    else:
      self._credentials_op = None

    if self._block_cache:
      self._block_cache_op = gen_gcs_config_ops.gcs_configure_block_cache(
          max_cache_size=self._block_cache.max_bytes,
          block_size=self._block_cache.block_size,
          max_staleness=self._block_cache.max_staleness)
    else:
      self._block_cache_op = None

  def after_create_session(self, session, coord):
    del coord
    if self._credentials_op:
      session.run(
          self._credentials_op,
          feed_dict={self._credentials_placeholder: self._credentials})
    if self._block_cache_op:
      session.run(self._block_cache_op)


def configure_gcs(session, credentials=None, block_cache=None, device=None):
  """Configures the GCS file system for a given a session.

  Warning: GCS `credentials` may be transmitted over the network unencrypted.
  Please ensure that the network is trusted before using this function. For
  users running code entirely within Google Cloud, your data is protected by
  encryption in between data centers. For more information, please take a look
  at https://cloud.google.com/security/encryption-in-transit/.

  Args:
    session: A `tf.Session` session that should be used to configure the GCS
      file system.
    credentials: [Optional.] A JSON string
    block_cache: [Optional.] A BlockCacheParams to configure the block cache .
    device: [Optional.] The device to place the configure ops.
  """

  def configure(credentials, block_cache):
    """Helper function to actually configure GCS."""
    if credentials:
      if isinstance(credentials, dict):
        credentials = json.dumps(credentials)
      placeholder = array_ops.placeholder(dtypes.string)
      op = gen_gcs_config_ops.gcs_configure_credentials(placeholder)
      session.run(op, feed_dict={placeholder: credentials})
    if block_cache:
      op = gen_gcs_config_ops.gcs_configure_block_cache(
          max_cache_size=block_cache.max_bytes,
          block_size=block_cache.block_size,
          max_staleness=block_cache.max_staleness)
      session.run(op)

  if device:
    with ops.device(device):
      return configure(credentials, block_cache)
  return configure(credentials, block_cache)


def configure_colab_session(session):
  """ConfigureColabSession configures the GCS file system in Colab.

  Args:
    session: A `tf.Session` session.
  """
  # Read from the application default credentials (adc).
  with open('/content/datalab/adc.json') as f:
    data = json.load(f)
  configure_gcs(session, credentials=data)
