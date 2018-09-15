# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Executable to estimate the volume of various sets of correlation matrices.

See correlation_matrix_volumes_lib.py for purpose and methodology.

Invocation example:
```
python correlation_matrix_volumes.py --num_samples 1e7
```

This will compute 10,000,000-sample confidence intervals for the
volumes of several sets of correlation matrices.  Which sets, and the
desired statistical significance, are hard-coded in this source file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

from absl import app
from absl import flags

from tensorflow.contrib.distributions.python.kernel_tests.util import correlation_matrix_volumes_lib as corr

FLAGS = flags.FLAGS

# Float to support giving the number of samples in scientific notation.
# The production run used for the LKJ test used 1e7 samples.
flags.DEFINE_float('num_samples', 1e4, 'Number of samples to use.')


def ctv_debatched(det_bounds, dim, num_samples, error_rate=1e-6, seed=42):
  # This wrapper undoes the batching in compute_true_volumes, because
  # apparently several 5x5x9x1e7 Tensors of float32 can strain RAM.
  bounds = {}
  for db in det_bounds:
    bounds[db] = corr.compute_true_volumes(
        [db], dim, num_samples, error_rate=error_rate, seed=seed)[db]
  return bounds


# The particular bounds in all three of these functions were chosen by
# a somewhat arbitrary walk through an empirical tradeoff, for the
# purpose of testing the LKJ distribution.  Setting the determinant
# bound lower
# - Covers more of the testee's sample space, and
# - Increases the probability that the rejection sampler will hit, thus
# - Decreases the relative error (at a fixed sample count) in the
#   rejection-based volume estimate;
# but also
# - Increases the variance of the estimator used in the LKJ test.
# This latter variance is also affected by the dimension and the
# tested concentration parameter, and can be compensated for with more
# compute (expensive) or a looser discrepancy limit (unsatisfying).
# The values here are the projection of the points in that test design
# space that ended up getting chosen.
def compute_3x3_volumes(num_samples):
  det_bounds = [0.01, 0.25, 0.3, 0.35, 0.4, 0.45]
  return ctv_debatched(
      det_bounds, 3, num_samples, error_rate=5e-7, seed=46)


def compute_4x4_volumes(num_samples):
  det_bounds = [0.01, 0.25, 0.3, 0.35, 0.4, 0.45]
  return ctv_debatched(
      det_bounds, 4, num_samples, error_rate=5e-7, seed=47)


def compute_5x5_volumes(num_samples):
  det_bounds = [0.01, 0.2, 0.25, 0.3, 0.35, 0.4]
  return ctv_debatched(
      det_bounds, 5, num_samples, error_rate=5e-7, seed=48)


def main(_):
  full_bounds = {}
  full_bounds[3] = compute_3x3_volumes(int(FLAGS.num_samples))
  full_bounds[4] = compute_4x4_volumes(int(FLAGS.num_samples))
  full_bounds[5] = compute_5x5_volumes(int(FLAGS.num_samples))
  pprint.pprint(full_bounds)

if __name__ == '__main__':
  app.run(main)
