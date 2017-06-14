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
"""Package for histogram compression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

# Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
# naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
# and then the long tail.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)


CompressedHistogramValue = collections.namedtuple('CompressedHistogramValue',
                                                  ['basis_point', 'value'])


def CompressHistogram(histo, bps=NORMAL_HISTOGRAM_BPS):
  """Creates fixed size histogram by adding compression to accumulated state.

  This routine transforms a histogram at a particular step by linearly
  interpolating its variable number of buckets to represent their cumulative
  weight at a constant number of compression points. This significantly reduces
  the size of the histogram and makes it suitable for a two-dimensional area
  plot where the output of this routine constitutes the ranges for a single x
  coordinate.

  Args:
    histo: A HistogramProto object.
    bps: Compression points represented in basis points, 1/100ths of a percent.
        Defaults to normal distribution.

  Returns:
    List of values for each basis point.
  """
  # See also: Histogram::Percentile() in core/lib/histogram/histogram.cc
  if not histo.num:
    return [CompressedHistogramValue(b, 0.0) for b in bps]
  bucket = np.array(histo.bucket)
  bucket_limit = list(histo.bucket_limit)
  weights = (bucket * bps[-1] / (bucket.sum() or 1.0)).cumsum()
  values = []
  j = 0
  while j < len(bps):
    i = np.searchsorted(weights, bps[j], side='right')
    while i < len(weights):
      cumsum = weights[i]
      cumsum_prev = weights[i - 1] if i > 0 else 0.0
      if cumsum == cumsum_prev:  # prevent remap divide by zero
        i += 1
        continue
      if not i or not cumsum_prev:
        lhs = histo.min
      else:
        lhs = max(bucket_limit[i - 1], histo.min)
      rhs = min(bucket_limit[i], histo.max)
      weight = _Remap(bps[j], cumsum_prev, cumsum, lhs, rhs)
      values.append(CompressedHistogramValue(bps[j], weight))
      j += 1
      break
    else:
      break
  while j < len(bps):
    values.append(CompressedHistogramValue(bps[j], histo.max))
    j += 1
  return values


def _Remap(x, x0, x1, y0, y1):
  """Linearly map from [x0, x1] unto [y0, y1]."""
  return y0 + (x - x0) * float(y1 - y0) / (x1 - x0)
