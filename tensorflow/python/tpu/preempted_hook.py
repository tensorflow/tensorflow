# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of the SessionRunHook for preemptible Cloud TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as _logging
import os
import threading
import time

from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook


class CloudTPUPreemptedHook(session_run_hook.SessionRunHook):
  """The SessionRunHook for preemptible Cloud TPUs.

  This is an implementation of SessionRunHook for the pre-emptible Google Cloud
  TPU service. It attempts to close the session if the TPU is preempted, and
  exits the coordinator process if the session cannot be closed.
  """

  def __init__(self, cluster):
    self._cluster = cluster

  def after_create_session(self, session, coord):
    if tpu_cluster_resolver.is_running_in_gce():
      self._tpu_poller = _TPUPollingThread(self._cluster, session)
      self._tpu_poller.start()

  def end(self, session):
    self._tpu_poller.stop()


class _TPUPollingThread(threading.Thread):
  """A thread that polls the state of a TPU node.

  When the node transitions into a TERMINAL state (PREEMPTED, TERMINATED)
  that's considered as not recoverable by the underlying infrastructure,
  it attempts to close the session, and exits the entire process if the
  session.close() stucks.
  """

  def __init__(self, cluster, session):
    super(_TPUPollingThread, self).__init__()

    self.daemon = True
    self._running = True
    self._session_closed = False
    self._cluster = cluster
    self._session = session
    self._interval = 30

    # Some of the Google API libraries are quite chatty, so disable them.
    for name in ['googleapiclient.discovery', 'oauth2client.client']:
      _logging.getLogger(name).setLevel(_logging.WARNING)

  def stop(self):
    self._running = False
    self._session_closed = True
    self.join()

  def run(self):
    if not tpu_cluster_resolver.is_running_in_gce():
      logging.warning(
          'TPUPollingThread is running in a non-GCE environment, exiting...')
      self._running = False
      return

    while self._running:
      recoverable = self._cluster._cloud_tpu_client.recoverable()  # pylint: disable=protected-access
      if not recoverable:
        logging.warning(
            'TPUPollingThread found TPU %s in state %s',
            self._cluster._tpu, self._cluster._cloud_tpu_client.state())  # pylint: disable=protected-access
        os._exit(1)  # pylint: disable=protected-access
      time.sleep(self._interval)
