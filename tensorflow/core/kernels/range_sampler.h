/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RANGE_SAMPLER_H_
#define TENSORFLOW_CORE_KERNELS_RANGE_SAMPLER_H_

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/weighted_picker.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Env;

// Abstract subclass for sampling from the set of non-negative integers
// [0, range)
class RangeSampler {
 public:
  explicit RangeSampler(int64 range) : range_(range) { CHECK_GT(range_, 0); }
  virtual ~RangeSampler();

  // Sample a single value
  virtual int64 Sample(random::SimplePhilox* rnd) const = 0;

  // The probability that a single call to Sample() returns the given value.
  // Assumes that value is in [0, range).  No range checking is done.
  virtual float Probability(int64 value) const = 0;

  // Fill "batch" with samples from the distribution.
  // If unique=true, then we re-pick each element until we get a
  // value distinct from all previously picked values in the batch.
  void SampleBatch(random::SimplePhilox* rnd, bool unique,
                   gtl::MutableArraySlice<int64> batch) const;

  // Fill "batch" with samples from the distribution, and report
  // "expected counts".
  //
  // The "expected count" of a value is an estimate of the expected
  // number of occurrences of the value in the batch returned by a
  // call to this function with the given parameters.  If unique=true,
  // the expected count is an inclusion probability.  For details on
  // this estimation, see the comment to "ExpectedCountHelper" in the
  // .cc file.
  //
  // Expected counts for the elements of the returned "batch" are reported
  // in the aligned array "batch_expected_count".
  //
  // The user can optionally provide "extras", containing values in the range.
  // The expected counts for the extras are reported in the aligned array
  // "extras_expected_count".
  //
  // "batch_expected_count" must have size equal to 0 or to the size of "batch".
  // "extras" and "extras_expected_count" must have equal size.
  void SampleBatchGetExpectedCount(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64> extras,
      gtl::MutableArraySlice<float> extras_expected_count) const;

  // Same as SampleBatchGetExpectedCount (see above), but with avoided values.
  // We repick to avoid all of the values in "avoided_values".
  // "avoided_values" is only supported with unique=true.  If
  // unique=false, then avoided_values must be empty.
  virtual void SampleBatchGetExpectedCountAvoid(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64> extras,
      gtl::MutableArraySlice<float> extras_expected_count,
      gtl::ArraySlice<int64> avoided_values) const;

  // Does this sampler need to be updated with values, e.g. UnigramSampler
  virtual bool NeedsUpdates() const { return false; }

  // Updates the underlying distribution
  virtual void Update(gtl::ArraySlice<int64> values) {
    LOG(FATAL) << "Update not supported for this sampler type.";
  }

  int64 range() { return range_; }

 protected:
  const int64 range_;
};

// An AllSampler only samples batches of size equal to range.
// It returns the entire range.
// It cannot sample single values.
class AllSampler : public RangeSampler {
 public:
  explicit AllSampler(int64 range);

  ~AllSampler() override {}

  int64 Sample(random::SimplePhilox* rnd) const override {
    LOG(FATAL) << "Should not be called";
    return 0;
  }

  float Probability(int64 value) const override {
    LOG(FATAL) << "Should not be called";
    return 0;
  }

  void SampleBatchGetExpectedCountAvoid(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64> extras,
      gtl::MutableArraySlice<float> extras_expected_count,
      gtl::ArraySlice<int64> avoided_values) const override;
};

class UniformSampler : public RangeSampler {
 public:
  explicit UniformSampler(int64 range);

  ~UniformSampler() override {}

  int64 Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64 value) const override;

 private:
  const float inv_range_;
};

class LogUniformSampler : public RangeSampler {
 public:
  explicit LogUniformSampler(int64 range);

  ~LogUniformSampler() override {}

  int64 Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64 value) const override;

 private:
  const double log_range_;
};

// Thread-unsafe unigram sampler
class ThreadUnsafeUnigramSampler : public RangeSampler {
 public:
  explicit ThreadUnsafeUnigramSampler(int64 range);
  ~ThreadUnsafeUnigramSampler() override {}

  int64 Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64 value) const override;

  bool NeedsUpdates() const override { return true; }
  void Update(gtl::ArraySlice<int64> values) override;

 private:
  random::WeightedPicker picker_;
};

// Thread-safe unigram sampler
class UnigramSampler : public RangeSampler {
 public:
  explicit UnigramSampler(int64 range);
  ~UnigramSampler() override {}

  int64 Sample(random::SimplePhilox* rnd) const override;

  float Probability(int64 value) const override;

  // Overriding at a high level results in far fewer lock acquisitions.
  void SampleBatchGetExpectedCountAvoid(
      random::SimplePhilox* rnd, bool unique,
      gtl::MutableArraySlice<int64> batch,
      gtl::MutableArraySlice<float> batch_expected_count,
      gtl::ArraySlice<int64> extras,
      gtl::MutableArraySlice<float> extras_expected_count,
      gtl::ArraySlice<int64> avoided_values) const override;

  bool NeedsUpdates() const override { return true; }
  void Update(gtl::ArraySlice<int64> values) override;

 private:
  ThreadUnsafeUnigramSampler unsafe_sampler_ TF_GUARDED_BY(mu_);
  mutable mutex mu_;
};

// A unigram sampler that uses a fixed unigram distribution read from a
// file or passed in as an in-memory array instead of building up the
// distribution from data on the fly. There is also an option to skew the
// distribution by applying a distortion power to the weights.
class FixedUnigramSampler : public RangeSampler {
 public:
  // The vocab_file is assumed to be a CSV, with the last entry of each row a
  // value representing the counts or probabilities for the corresponding ID.
  FixedUnigramSampler(Env* env, int64 range, const string& vocab_file,
                      float distortion, int32 num_reserved_ids,
                      int32 num_shards, int32 shard);

  FixedUnigramSampler(int64 range, const std::vector<float>& unigrams,
                      float distortion, int32 num_reserved_ids,
                      int32 num_shards, int32 shard);

  float Probability(int64 value) const override;

  int64 Sample(random::SimplePhilox* rnd) const override;

 private:
  // Underlying distribution sampler.
  std::unique_ptr<random::DistributionSampler> dist_sampler_;
  // Weights for individual samples. The probability of a sample i is defined
  // as weights_.at(i) / total_weight_.
  std::vector<float> weights_;
  // The total weights of all samples.
  float total_weight_;
  // Sharding information of the sampler. The whole vocabulary is sharded
  // into num_shards_ smaller ranges and each sampler is responsible for one
  // such smaller range, identified by the shard number.
  int32 num_shards_;
  int32 shard_;

  // Fill the sampler with the appropriate number of reserved IDs.
  void FillReservedIds(int32 num_reserved_ids);
  // Load IDs to sample from a CSV file. It is assumed that the last item of
  // each row contains a count or probability for the corresponding ID.
  Status LoadFromFile(Env* env, const string& vocab_file, float distortion);
  // Load from an in-memory array.
  void LoadFromUnigrams(const std::vector<float>& unigrams, float distortion);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RANGE_SAMPLER_H_
