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
# ===================================================================

"""TPU constant configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib

# pylint: enable=protected-access
_TF_CONFIG_ENV = run_config_lib._TF_CONFIG_ENV
_SERVICE_KEY = run_config_lib._SERVICE_KEY
_TPU_WORKER_JOB_NAME = 'tpu_worker_job_name'

_INITIAL_LOSS = 1e7
_ZERO_LOSS = 0.
_TPU_ESTIMATOR = 'tpu_estimator'
_ITERATIONS_PER_LOOP_VAR = 'iterations_per_loop'
_CROSS_REPLICA_SUM_OP = 'CrossReplicaSum'

# TODO(b/65703635): Flip the value and remove all dead code.
_WRAP_INPUT_FN_WHILE_LOOP = False

_DEFAULT_JOB_NAME = 'tpu_worker'
_DEFAULT_COORDINATOR_JOB_NAME = 'coordinator'
_LOCAL_MASTERS = ('', 'local')

_DEFAULT_NUMBER_OF_SHARDS = 1
_DEFAULT_SHARD_DIMENSION = 0

# Operations that indicate some error in the users graph, e.g. a placeholder
# that's introduced outside of the infeed.
_BLACKLISTED_OPS = set([
    "Placeholder",
])

# These operations will currently fail to compile, but we should be able to
# support them eventually via CPU offload or extending our operation set.
_NOT_IMPLEMENTED_OPS = set([
    "AudioSummary",
    "AudioSummaryV2",
    "HistogramSummary",
    "ImageSummary",
    "MergeSummary",
    "Print",
    "ScalarSummary",
    "TensorSummary",
    "TensorSummaryV2",
    ])

_MAX_WARNING_LINES = 5

_TPU_REPLICATE_ATTR = "_tpu_replicate"

