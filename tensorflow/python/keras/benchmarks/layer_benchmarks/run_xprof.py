# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import time
import uuid

from tensorflow.python.profiler import profiler_v2 as profiler

def run_with_xprof(self, func, num_iters_xprof=100, enable_python_trace=True,
                   logdir='/tmp/layer_benchmark_xprof/'):
  suid = str(uuid.uuid4())
  if enable_python_trace:
    options = profiler.ProfilerOptions(python_tracer_level=1)
    logdir = os.path.join(logdir, str(uuid.uuid4()) + "_with_python")
  else:
    options = profiler.ProfilerOptions(python_tracer_level=0)
    logdir = os.path.join(logdir, suid)

  start = time.time()
  with profiler.Profile(logdir, options):
    for _ in range(num_iters_xprof):
      func()
  total_time = time.time() - start
  us_per_example = float("{0:.3f}".format(total_time * 1e6 / num_iters_xprof))
  return logdir, us_per_example
