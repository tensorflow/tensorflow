/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/costs/robust_stats.h"
#include <algorithm>
#include <cmath>
#include <utility>

namespace tensorflow {
namespace grappler {

// Given a sorted vector of values, calculate the median.
// Returns 0 for an empty vector.  Does not verify sortedness.
static double SortedMedian(const std::vector<double> &values) {
  const int n = values.size();
  if (n == 0) return 0.0;
  if (n & 1) {
    return values[n / 2];
  } else {
    return (values[n / 2] + values[n / 2 - 1]) / 2.0;
  }
}

// Given a vector of values (sorted or not), calculate the median.
static double Median(std::vector<double> &&values) {
  const size_t n = values.size();
  if (n == 0) return 0;
  const auto middle = values.begin() + (n / 2);
  // Put the middle value in its place.
  std::nth_element(values.begin(), middle, values.end());
  if (n & 1) {
    return *middle;
  }
  // Return the average of the two elements, the max_element lower than
  // *middle is found between begin and middle as a post-cond of
  // nth_element.
  const auto lower_middle = std::max_element(values.begin(), middle);
  // Preventing overflow. We know that '*lower_middle <= *middle'.
  // If both are on opposite sides of zero, the sum won't overflow, otherwise
  // the difference won't overflow.
  if (*lower_middle <= 0 && *middle >= 0) {
    return (*lower_middle + *middle) / 2;
  }
  return *lower_middle + (*middle - *lower_middle) / 2;
}

// Given a set of values, calculates the scaled Median Absolute Deviation (a
// robust approximation to the standard deviation).  This is calculated as the
// median of the absolute deviations from the median, scaled by 1.4826.  Its
// advantage over the standard deviation is that it is not (as) affected by
// outlier values.  Returns a pair<median, mad>.
static std::pair<double, double> ScaledMedianAbsoluteDeviation(
    const std::vector<double> &sorted_values) {
  double median = SortedMedian(sorted_values);

  // Next, we calculate the absolute deviations from the median,
  // find the median of the resulting data, and scale by 1.4826.
  std::vector<double> deviations;
  deviations.reserve(sorted_values.size());
  for (double d : sorted_values) {
    deviations.push_back(std::abs(d - median));
  }
  double mad = Median(std::move(deviations)) * 1.4826;
  return std::pair<double, double>(median, mad);
}

RobustStats::RobustStats(const std::vector<double> &values)
    : RobustStats(std::vector<double>(values)) {}

RobustStats::RobustStats(std::vector<double> &&values) {
  std::sort(values.begin(), values.end());
  lo_ = values[0];
  hi_ = values.back();
  HuberMAD(values);
}

// Computes an updated mean using Huber's weighting function (values beyond
// the margin are weighted by margin / abs(value - mean).
double UpdateHuberMean(const std::vector<double> &sorted_values, double mean,
                       double margin) {
  int num_within = 0;
  double sum = 0.0;

  for (double d : sorted_values) {
    if (d < mean - margin) {
      sum -= margin;
    } else if (d > mean + margin) {
      sum += margin;
    } else {
      sum += d;
      ++num_within;
    }
  }

  // It is possible, for a set with an interquartile distance of 0, i.e., with
  // more than half of the values at the median, to encounter the case where
  // the Huber mean drifts slightly off the median and there are no values
  // within the margin.  In that case, just return the old mean, and the caller
  // will quit.
  if (num_within > 0) {
    return sum / num_within;
  } else {
    return mean;
  }
}

// Given a list of values, this approximates the stddev using the MAD and then
// uses it to compute a Huber robust mean (sandwich mean).  A margin of
// c*stddev is defined around the current mean, and values are weighted by
// margin / abs(value - mean) if outside the margin, or 1 if inside.  This
// computes the mean iteratively, because each time it changes the margin
// shifts a bit.  It typically settles very quickly, but it's possible for it
// to be unstable.  We limit it to 10 iterations.
//
void RobustStats::HuberMAD(const std::vector<double> &sorted_values) {
  const std::pair<double, double> median_mad =
      ScaledMedianAbsoluteDeviation(sorted_values);
  mean_ = median_mad.first;
  stddev_ = median_mad.second;

  // c = 1.345 is the commonly used cutoff with 95% efficiency at the normal.
  // We're using c = 1.5 to be a little more conservative, and because that's
  // the default in S-plus.
  // TODO(dehnert): Specialize Stats for integral types so we don't implement
  // methods that don't make sense.
  const double c = 1.5;
  const double margin = c * stddev_;

  // Iterate 10 times, or until the Huber mean stabilizes.
  // If the margin is zero, we don't want mean to drift from the median.
  if (margin > 0.0) {
    for (int k = 0; k < 10; ++k) {
      double old_mean = mean_;
      mean_ = UpdateHuberMean(sorted_values, mean_, margin);
      if (mean_ == old_mean) break;
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow
