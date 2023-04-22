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
"""Stub file to maintain backwards compatibility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import,unused-import,redefined-builtin
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import *
# used by tests
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _clone_export_output_with_tensors
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _create_global_step
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _export_output_to_tensors
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _get_scaffold
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _Inputs
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _ITERATIONS_PER_LOOP_VAR
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _TPU_ENQUEUE_OPS
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _TPU_ESTIMATOR
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import _TPU_TRAIN_OP
# pylint: enable=wildcard-import,unused-import,redefined-builtin
