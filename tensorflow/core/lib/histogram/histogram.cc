/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/histogram/histogram.h"
#include <float.h>
#include <math.h>
#include <vector>
#include "tensorflow/core/framework/summary.pb.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace histogram {

static std::vector<double>* InitDefaultBucketsInner() {
  std::vector<double> buckets;
  std::vector<double> neg_buckets;
  // Make buckets whose range grows by 10% starting at 1.0e-12 up to 1.0e20
  double v = 1.0e-12;
  while (v < 1.0e20) {
    buckets.push_back(v);
    neg_buckets.push_back(-v);
    v *= 1.1;
  }
  buckets.push_back(DBL_MAX);
  neg_buckets.push_back(-DBL_MAX);
  std::reverse(neg_buckets.begin(), neg_buckets.end());
  std::vector<double>* result = new std::vector<double>;
  result->insert(result->end(), neg_buckets.begin(), neg_buckets.end());
  result->push_back(0.0);
  result->insert(result->end(), buckets.begin(), buckets.end());
  return result;
}

static gtl::ArraySlice<double> InitDefaultBuckets() {
  static std::vector<double>* default_bucket_limits = InitDefaultBucketsInner();
  return *default_bucket_limits;
}

Histogram::Histogram() : bucket_limits_(InitDefaultBuckets()) { Clear(); }

// Create a histogram with a custom set of bucket limits,
// specified in "custom_buckets[0..custom_buckets.size()-1]"
Histogram::Histogram(gtl::ArraySlice<double> custom_bucket_limits)
    : custom_bucket_limits_(custom_bucket_limits.begin(),
                            custom_bucket_limits.end()),
      bucket_limits_(custom_bucket_limits_) {
#ifndef NDEBUG
  DCHECK_GT(bucket_limits_.size(), size_t{0});
  // Verify that the bucket boundaries are strictly increasing
  for (size_t i = 1; i < bucket_limits_.size(); i++) {
    DCHECK_GT(bucket_limits_[i], bucket_limits_[i - 1]);
  }
#endif
  Clear();
}

bool Histogram::DecodeFromProto(const HistogramProto& proto) {
  if ((proto.bucket_size() != proto.bucket_limit_size()) ||
      (proto.bucket_size() == 0)) {
    return false;
  }
  min_ = proto.min();
  max_ = proto.max();
  num_ = proto.num();
  sum_ = proto.sum();
  sum_squares_ = proto.sum_squares();
  custom_bucket_limits_.clear();
  custom_bucket_limits_.insert(custom_bucket_limits_.end(),
                               proto.bucket_limit().begin(),
                               proto.bucket_limit().end());
  bucket_limits_ = custom_bucket_limits_;
  buckets_.clear();
  buckets_.insert(buckets_.end(), proto.bucket().begin(), proto.bucket().end());
  return true;
}

void Histogram::Clear() {
  min_ = bucket_limits_[bucket_limits_.size() - 1];
  max_ = -DBL_MAX;
  num_ = 0;
  sum_ = 0;
  sum_squares_ = 0;
  buckets_.resize(bucket_limits_.size());
  for (size_t i = 0; i < bucket_limits_.size(); i++) {
    buckets_[i] = 0;
  }
}

void Histogram::Add(double value) {
  int b =
      std::upper_bound(bucket_limits_.begin(), bucket_limits_.end(), value) -
      bucket_limits_.begin();

  buckets_[b] += 1.0;
  if (min_ > value) min_ = value;
  if (max_ < value) max_ = value;
  num_++;
  sum_ += value;
  sum_squares_ += (value * value);
}

double Histogram::Median() const { return Percentile(50.0); }

// Linearly map the variable x from [x0, x1] unto [y0, y1]
double Histogram::Remap(double x, double x0, double x1, double y0,
                        double y1) const {
  return y0 + (x - x0) / (x1 - x0) * (y1 - y0);
}

// Pick tight left-hand-side and right-hand-side bounds and then
// interpolate a histogram value at percentile p
double Histogram::Percentile(double p) const {
  if (num_ == 0.0) return 0.0;

  double threshold = num_ * (p / 100.0);
  double cumsum_prev = 0;
  for (size_t i = 0; i < buckets_.size(); i++) {
    double cumsum = cumsum_prev + buckets_[i];

    // Find the first bucket whose cumsum >= threshold
    if (cumsum >= threshold) {
      // Prevent divide by 0 in remap which happens if cumsum == cumsum_prev
      // This should only get hit when p == 0, cumsum == 0, and cumsum_prev == 0
      if (cumsum == cumsum_prev) {
        continue;
      }

      // Calculate the lower bound of interpolation
      double lhs = (i == 0 || cumsum_prev == 0) ? min_ : bucket_limits_[i - 1];
      lhs = std::max(lhs, min_);

      // Calculate the upper bound of interpolation
      double rhs = bucket_limits_[i];
      rhs = std::min(rhs, max_);

      double weight = Remap(threshold, cumsum_prev, cumsum, lhs, rhs);
      return weight;
    }

    cumsum_prev = cumsum;
  }
  return max_;
}

double Histogram::Average() const {
  if (num_ == 0.0) return 0;
  return sum_ / num_;
}

double Histogram::StandardDeviation() const {
  if (num_ == 0.0) return 0;
  double variance = (sum_squares_ * num_ - sum_ * sum_) / (num_ * num_);
  return sqrt(variance);
}

std::string Histogram::ToString() const {
  std::string r;
  char buf[200];
  snprintf(buf, sizeof(buf), "Count: %.0f  Average: %.4f  StdDev: %.2f\n", num_,
           Average(), StandardDeviation());
  r.append(buf);
  snprintf(buf, sizeof(buf), "Min: %.4f  Median: %.4f  Max: %.4f\n",
           (num_ == 0.0 ? 0.0 : min_), Median(), max_);
  r.append(buf);
  r.append("------------------------------------------------------\n");
  const double mult = num_ > 0 ? 100.0 / num_ : 0.0;
  double sum = 0;
  for (size_t b = 0; b < buckets_.size(); b++) {
    if (buckets_[b] <= 0.0) continue;
    sum += buckets_[b];
    snprintf(buf, sizeof(buf), "[ %10.2g, %10.2g ) %7.0f %7.3f%% %7.3f%% ",
             ((b == 0) ? -DBL_MAX : bucket_limits_[b - 1]),  // left
             bucket_limits_[b],                              // right
             buckets_[b],                                    // count
             mult * buckets_[b],                             // percentage
             mult * sum);                                    // cum percentage
    r.append(buf);

    // Add hash marks based on percentage; 20 marks for 100%.
    int marks = static_cast<int>(20 * (buckets_[b] / num_) + 0.5);
    r.append(marks, '#');
    r.push_back('\n');
  }
  return r;
}

void Histogram::EncodeToProto(HistogramProto* proto,
                              bool preserve_zero_buckets) const {
  proto->Clear();
  proto->set_min(min_);
  proto->set_max(max_);
  proto->set_num(num_);
  proto->set_sum(sum_);
  proto->set_sum_squares(sum_squares_);
  for (size_t i = 0; i < buckets_.size();) {
    double end = bucket_limits_[i];
    double count = buckets_[i];
    i++;
    if (!preserve_zero_buckets && count <= 0.0) {
      // Find run of empty buckets and collapse them into one
      while (i < buckets_.size() && buckets_[i] <= 0.0) {
        end = bucket_limits_[i];
        count = buckets_[i];
        i++;
      }
    }
    proto->add_bucket_limit(end);
    proto->add_bucket(count);
  }
  if (proto->bucket_size() == 0.0) {
    // It's easier when we restore if we always have at least one bucket entry
    proto->add_bucket_limit(DBL_MAX);
    proto->add_bucket(0.0);
  }
}

// ThreadSafeHistogram implementation.
bool ThreadSafeHistogram::DecodeFromProto(const HistogramProto& proto) {
  mutex_lock l(mu_);
  return histogram_.DecodeFromProto(proto);
}

void ThreadSafeHistogram::Clear() {
  mutex_lock l(mu_);
  histogram_.Clear();
}

void ThreadSafeHistogram::Add(double value) {
  mutex_lock l(mu_);
  histogram_.Add(value);
}

void ThreadSafeHistogram::EncodeToProto(HistogramProto* proto,
                                        bool preserve_zero_buckets) const {
  mutex_lock l(mu_);
  histogram_.EncodeToProto(proto, preserve_zero_buckets);
}

double ThreadSafeHistogram::Median() const {
  mutex_lock l(mu_);
  return histogram_.Median();
}

double ThreadSafeHistogram::Percentile(double p) const {
  mutex_lock l(mu_);
  return histogram_.Percentile(p);
}

double ThreadSafeHistogram::Average() const {
  mutex_lock l(mu_);
  return histogram_.Average();
}

double ThreadSafeHistogram::StandardDeviation() const {
  mutex_lock l(mu_);
  return histogram_.StandardDeviation();
}

std::string ThreadSafeHistogram::ToString() const {
  mutex_lock l(mu_);
  return histogram_.ToString();
}

}  // namespace histogram
}  // namespace tensorflow
