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
"""Environment configuration object for Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TaskType(object):
  MASTER = 'master'
  PS = 'ps'
  WORKER = 'worker'


class RunConfig(object):
  """This class specifies the configurations for an `Estimator` run."""

  @property
  def cluster_spec(self):
    return None

  @property
  def evaluation_master(self):
    return ''

  @property
  def is_chief(self):
    return True

  @property
  def master(self):
    return ''

  @property
  def num_ps_replicas(self):
    return 0

  @property
  def num_worker_replicas(self):
    return 1

  @property
  def task_id(self):
    return 0

  @property
  def task_type(self):
    return TaskType.WORKER

  @property
  def tf_random_seed(self):
    return 1

  @property
  def save_summary_steps(self):
    return 100

  @property
  def save_checkpoints_secs(self):
    return 600

  @property
  def save_checkpoints_steps(self):
    return None

  @property
  def keep_checkpoint_max(self):
    return 5

  @property
  def keep_checkpoint_every_n_hours(self):
    return 10000
