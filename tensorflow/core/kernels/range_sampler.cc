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

#include "tensorflow/core/kernels/range_sampler.h"

#include <cmath>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using gtl::ArraySlice;
using gtl::MutableArraySlice;

RangeSampler::~RangeSampler() {}

void RangeSampler::SampleBatch(random::SimplePhilox* rnd, bool unique,
                               absl::Span<int64_t> batch) const {
  SampleBatchGetExpectedCount(rnd, unique, batch, absl::Span<float>(),
                              absl::Span<const int64_t>(), absl::Span<float>());
}

void RangeSampler::SampleBatchGetExpectedCount(
    random::SimplePhilox* rnd, bool unique, absl::Span<int64_t> batch,
    absl::Span<float> batch_expected_count, absl::Span<const int64_t> extras,
    absl::Span<float> extras_expected_count) const {
  SampleBatchGetExpectedCountAvoid(rnd, unique, batch, batch_expected_count,
                                   extras, extras_expected_count,
                                   absl::Span<const int64_t>());
}

namespace {

// Approximates the expected count of a value in the output of SampleBatch.
//
// If unique=false, then this is (Probability(value) * batch_size)
//
// We use batch_size and num_tries, where num_tries is the observed number of
// tries it took to get batch_size unique values.
//
// Assuming (falsely) that the number of tries to get a batch of batch_size
// distinct values is _always_ num_tries, the probability that the value
// is in a batch is (1 - (1-p)^num_tries)
static float ExpectedCountHelper(float p, int batch_size, int num_tries) {
  if (num_tries == batch_size) {
    // This shortcut will always be taken if unique=false
    return p * batch_size;
  }
  // numerically stable version of (1 - (1-p)^num_tries)
  return -std::expm1(num_tries * std::log1p(-p));
}

}  // namespace

void RangeSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, absl::Span<int64_t> batch,
    absl::Span<float> batch_expected_count, absl::Span<const int64_t> extras,
    absl::Span<float> extras_expected_count,
    absl::Span<const int64_t> avoided_values) const {
  const int batch_size = batch.size();
  int num_tries;

  if (unique) {
    CHECK_LE(static_cast<int64_t>(batch_size + avoided_values.size()), range_);
    std::unordered_set<int64_t> used(batch_size);
    used.insert(avoided_values.begin(), avoided_values.end());
    int num_picked = 0;
    num_tries = 0;
    while (num_picked < batch_size) {
      num_tries++;
      CHECK_LT(num_tries, kint32max);
      int64_t value = Sample(rnd);
      if (gtl::InsertIfNotPresent(&used, value)) {
        batch[num_picked++] = value;
      }
    }
  } else {
    CHECK_EQ(avoided_values.size(), size_t{0})
        << "avoided_values only supported with unique=true";
    for (int i = 0; i < batch_size; i++) {
      batch[i] = Sample(rnd);
    }
    num_tries = batch_size;
  }
  // Compute the expected counts of the batch and the extra values
  if (!batch_expected_count.empty()) {
    CHECK_EQ(batch_size, batch_expected_count.size());
    for (int i = 0; i < batch_size; i++) {
      batch_expected_count[i] =
          ExpectedCountHelper(Probability(batch[i]), batch_size, num_tries);
    }
  }
  CHECK_EQ(extras.size(), extras_expected_count.size());
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] =
        ExpectedCountHelper(Probability(extras[i]), batch_size, num_tries);
  }
}

AllSampler::AllSampler(int64_t range) : RangeSampler(range) {}

void AllSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, absl::Span<int64_t> batch,
    absl::Span<float> batch_expected_count, absl::Span<const int64_t> extras,
    absl::Span<float> extras_expected_count,
    absl::Span<const int64_t> avoided_values) const {
  const int batch_size = batch.size();
  CHECK_EQ(range_, batch_size);
  for (int i = 0; i < batch_size; i++) {
    batch[i] = i;
  }
  if (!batch_expected_count.empty()) {
    CHECK_EQ(batch_size, batch_expected_count.size());
    for (int i = 0; i < batch_size; i++) {
      batch_expected_count[i] = 1;
    }
  }
  CHECK_EQ(size_t{0}, avoided_values.size());
  CHECK_EQ(extras.size(), extras_expected_count.size());
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] = 1;
  }
}

UniformSampler::UniformSampler(int64_t range)
    : RangeSampler(range), inv_range_(1.0 / range) {}

int64_t UniformSampler::Sample(random::SimplePhilox* rnd) const {
  return rnd->Uniform64(range_);
}

float UniformSampler::Probability(int64_t value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64_t range)
    : RangeSampler(range), log_range_(log1p(range)) {}

int64_t LogUniformSampler::Sample(random::SimplePhilox* rnd) const {
  const int64_t value =
      static_cast<int64_t>(exp(rnd->RandDouble() * log_range_)) - 1;
  DCHECK_GE(value, 0);
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.  In practice this case
  // happens never regardless of the value of range_, including and up to
  // DBL_MAX.  But we include it as a guarantee of the function's output.
  return value % range_;
}

float LogUniformSampler::Probability(int64_t value) const {
  // value is returned iff the call to UniformDouble(log_range_) in the
  // Sample() function returns a value between log(value + 1)
  // and log(value + 2).   The probability of this is:
  // (log(value + 2) - log(value + 1)) / log_range
  // To avoid two calls to log(), we compute this as follows:
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

ThreadUnsafeUnigramSampler::ThreadUnsafeUnigramSampler(int64_t range)
    : RangeSampler(range), picker_(range) {
  CHECK_LT(range, kint32max);
}

int64_t ThreadUnsafeUnigramSampler::Sample(random::SimplePhilox* rnd) const {
  return picker_.Pick(rnd);
}

float ThreadUnsafeUnigramSampler::Probability(int64_t value) const {
  return static_cast<float>(picker_.get_weight(value)) / picker_.total_weight();
}

void ThreadUnsafeUnigramSampler::Update(absl::Span<const int64_t> values) {
  int num_updates = std::min(static_cast<int>(values.size()),
                             kint32max - picker_.total_weight());
  for (int i = 0; i < num_updates; i++) {
    const int64_t value = values[i];
    picker_.set_weight(value, picker_.get_weight(value) + 1);
  }
}

// Thread-safe unigram sampler
UnigramSampler::UnigramSampler(int64_t range)
    : RangeSampler(range), unsafe_sampler_(range) {
  CHECK_LT(range, kint32max);
}

int64_t UnigramSampler::Sample(random::SimplePhilox* rnd) const {
  tf_shared_lock lock(mu_);
  return unsafe_sampler_.Sample(rnd);
}

float UnigramSampler::Probability(int64_t value) const {
  tf_shared_lock lock(mu_);
  return unsafe_sampler_.Probability(value);
}

// Overriding at a high level results in far fewer lock acquisitions.
void UnigramSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, absl::Span<int64_t> batch,
    absl::Span<float> batch_expected_count, absl::Span<const int64_t> extras,
    absl::Span<float> extras_expected_count,
    absl::Span<const int64_t> avoided_values) const {
  tf_shared_lock lock(mu_);
  unsafe_sampler_.SampleBatchGetExpectedCountAvoid(
      rnd, unique, batch, batch_expected_count, extras, extras_expected_count,
      avoided_values);
}

void UnigramSampler::Update(absl::Span<const int64_t> values) {
  mutex_lock lock(mu_);
  unsafe_sampler_.Update(values);
}

FixedUnigramSampler::FixedUnigramSampler(int64_t range, float distortion,
                                         int32_t num_reserved_ids,
                                         int32_t num_shards, int32_t shard)
    : RangeSampler(range),
      total_weight_(0.0),
      num_shards_(num_shards),
      shard_(shard),
      distortion_(distortion) {
  FillReservedIds(num_reserved_ids);
}

absl::Status FixedUnigramSampler::SetDistributionSampler(
    Env* env, const string& vocab_file) {
  TF_RETURN_IF_ERROR(LoadFromFile(env, vocab_file, distortion_));
  if (!TF_PREDICT_TRUE(FixedUnigramSampler::range() == weights_.size()))
    return (errors::InvalidArgument("range is ", FixedUnigramSampler::range(),
                                    " must be equal to weights size  ",
                                    weights_.size()));
  dist_sampler_.reset(new random::DistributionSampler(weights_));
  return absl::OkStatus();
}

absl::Status FixedUnigramSampler::SetDistributionSampler(
    const std::vector<float>& unigrams) {
  LoadFromUnigrams(unigrams, distortion_);
  if (!TF_PREDICT_TRUE(FixedUnigramSampler::range() == weights_.size()))
    return (errors::InvalidArgument("range is ", FixedUnigramSampler::range(),
                                    " must be equal to weights size  ",
                                    weights_.size()));
  dist_sampler_.reset(new random::DistributionSampler(weights_));
  return absl::OkStatus();
}

float FixedUnigramSampler::Probability(int64_t value) const {
  if (value < 0 || static_cast<size_t>(value) >= weights_.size()) {
    return 0.0;
  }
  return weights_.at(value) / total_weight_;
}

int64_t FixedUnigramSampler::Sample(random::SimplePhilox* rnd) const {
  return dist_sampler_->Sample(rnd);
}

void FixedUnigramSampler::FillReservedIds(int32_t num_reserved_ids) {
  for (int32_t word_id = 0; word_id < num_reserved_ids; ++word_id) {
    if (word_id % num_shards_ == shard_) weights_.push_back(0.0);
  }
}

absl::Status FixedUnigramSampler::LoadFromFile(Env* env,
                                               const string& vocab_file,
                                               float distortion) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(vocab_file, &file));

  io::InputBuffer in(file.get(), 262144 /*bytes*/);
  string line;
  int32_t word_id = weights_.size();
  while (in.ReadLine(&line).ok()) {
    // The vocabulary file should be in csv like format, with the last
    // field the weight associated with the word.
    std::vector<string> cols = str_util::Split(line, ',');
    if (cols.empty()) continue;
    // Skip entries that do not belong to this shard.
    if (word_id % num_shards_ == shard_) {
      float w = 0.0;
      if (!strings::safe_strtof(cols.at(cols.size() - 1), &w)) {
        return errors::InvalidArgument("Wrong vocabulary format at line: ",
                                       line);
      }
      w = std::pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
  return absl::OkStatus();
}

void FixedUnigramSampler::LoadFromUnigrams(const std::vector<float>& unigrams,
                                           float distortion) {
  int32_t word_id = weights_.size();
  for (float w : unigrams) {
    // Skip entries that do not belong to this shard.
    if (word_id % num_shards_ == shard_) {
      w = std::pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
}

}  // namespace tensorflow
