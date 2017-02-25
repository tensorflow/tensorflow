# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for running, waiting and stopping benchmark jobs on kubernetes.

Functions in this file assume kubernetes jobs have 'name-prefix' and 'job'
selectors set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import subprocess
import time


_KUBECTL = 'kubectl'
WAIT_PERIOD_SECONDS = 20


class TimeoutError(Exception):
  pass


def _WaitUntil(timeout, predicate, *args):
  start_time = time.time()
  while time.time() - start_time < timeout:
    time.sleep(WAIT_PERIOD_SECONDS)
    if predicate(*args):
      return True
  return False


def _GetPodNames(pod_name_prefix, job_name=None):
  """Get pod names based on the pod_name_prefix and job_name.

  Args:
    pod_name_prefix: value of 'name-prefix' selector.
    job_name: value of 'job' selector. If None, pod names will be
      selected only based on 'name-prefix' selector.

  Returns:
    List of pod names.
  """
  pod_list_command = [
      _KUBECTL, 'get', 'pods', '-o', 'name',
      '-l', _GetJobSelector(pod_name_prefix, job_name)]
  logging.info('Command to get pod names: %s', ' '.join(pod_list_command))
  output = subprocess.check_output(pod_list_command, universal_newlines=True)
  pod_names = [name for name in output.strip().split('\n') if name]
  logging.info('Pod names: "%s"', ','.join(pod_names))
  return pod_names


def CreatePods(pod_name, yaml_file):
  """Creates pods based on the given kubernetes config.

  Args:
    pod_name: 'name-prefix' selector for the pods.
    yaml_file: kubernetes yaml config.

  Raises:
    TimeoutError: if jobs didn't come up for a long time.
  """
  command = [_KUBECTL, 'create', '--filename=%s' % yaml_file]
  logging.info('Creating pods: %s', subprocess.list2cmdline(command))
  subprocess.check_call(command)

  if not _WaitUntil(100, _GetPodNames, pod_name):
    raise TimeoutError(
        'Timed out waiting for %s pod to come up.' % pod_name)


def DeletePods(pod_name, yaml_file):
  """Deletes pods based on the given kubernetes config.

  Args:
    pod_name: 'name-prefix' selector for the pods.
    yaml_file: kubernetes yaml config.

  Raises:
    TimeoutError: if jobs didn't terminate for a long time.
  """
  command = [_KUBECTL, 'delete', '--filename=%s' % yaml_file]
  logging.info('Deleting pods: %s', ' '.join(command))
  subprocess.call(command)

  def CheckPodsAreTerminated():
    return not _GetPodNames(pod_name)
  if not _WaitUntil(100, CheckPodsAreTerminated):
    raise TimeoutError(
        'Timed out waiting for %s pod to terminate.' % pod_name)


def _GetJobSelector(pod_name_prefix, job_name=None):
  selector = 'name-prefix in (%s)' % pod_name_prefix
  if job_name:
    selector += ',job in (%s)' % job_name
  return selector


def WaitForCompletion(pod_name_prefix, job_name='worker', timeout=2*60*60):
  """Waits until jobs matching pod_name and job_name are terminated.

  Args:
    pod_name_prefix: value of 'name-prefix' selector.
    job_name: value of 'job' selector.
    timeout: how long to wait for jobs to terminate before timing out.

  Returns:
    True if jobs terminated with success, False otherwise.

  Raises:
    TimeoutError: if jobs haven't terminated after timeout.
    ValueError: if we couldn't find jobs matching pod_name and job_name.
  """
  # Jsonpath that selects comma-separated exit codes (followed by extra comma
  # at the end).
  # If a job doesn't have an exit code yet, empty string will be returned
  # instead. For ex. output for 2 jobs where one is missing an exit code
  # and the other one has an exit code of 0 would look like: ,0,
  last_state_query = (
      'jsonpath=\'{range .items[*]}'
      '{.status.containerStatuses[?(@.lastState.terminated)]'
      '.lastState.terminated.exitCode},{end}\'')
  status_command = [
      _KUBECTL, 'get', '-o', last_state_query,
      'pods', '-l', _GetJobSelector(pod_name_prefix, job_name)
  ]

  exit_codes = []
  start_time = time.time()
  while time.time() - start_time < timeout:
    # Output of check_output is a string that starts and ends with '.
    output = subprocess.check_output(
        status_command, universal_newlines=True).strip('\'')
    logging.debug('Pod status: %s', output)
    if not output:
      raise ValueError(
          'Query did not match any data. Query: %s' % ' '.join(status_command))
    # Output will end with an extra comma. So, we remove it before splitting.
    exit_codes = output[:-1].split(',')
    if '' not in exit_codes:  # fetched all exit codes
      break
    time.sleep(WAIT_PERIOD_SECONDS)

  if '' in exit_codes:
    raise TimeoutError(
        'Timed out waiting for %s %s jobs to finish.' %
        (pod_name_prefix, job_name))
  _PrintLogs(pod_name_prefix, job_name)

  failed_job_count = sum(code != '0' for code in exit_codes)
  if failed_job_count > 0:
    logging.error('%d out of %d jobs failed. Exit codes: %s',
                  failed_job_count, len(exit_codes), ','.join(exit_codes))
    return False
  return True


def _PrintLogs(pod_name_prefix, job_name):
  """Prints pod logs.

  If a pod has been restarted, prints logs from previous run. Otherwise,
  prints the logs from current run. We print logs for pods selected
  based on pod_name_prefix and job_name.

  Args:
    pod_name_prefix: value of 'name-prefix' selector.
    job_name: value of 'job' selector.
  """
  for pod_name in _GetPodNames(pod_name_prefix, job_name):
    try:
      # Get previous logs.
      logs_command = [_KUBECTL, 'logs', '-p', pod_name]
      logging.info('Command to get logs: %s', ' '.join(logs_command))
      output = subprocess.check_output(logs_command, universal_newlines=True)
    except subprocess.CalledProcessError:
      # We couldn't get previous logs, so we will try to get current logs.
      logs_command = [_KUBECTL, 'logs', pod_name]
      logging.info('Command to get logs: %s', ' '.join(logs_command))
      output = subprocess.check_output(logs_command, universal_newlines=True)
    print('%s logs:' % pod_name)
    print(output)
