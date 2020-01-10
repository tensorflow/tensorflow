#include "tensorflow/core/kernels/range_sampler.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

using gtl::ArraySlice;
using gtl::MutableArraySlice;

RangeSampler::~RangeSampler() {}

void RangeSampler::SampleBatch(random::SimplePhilox* rnd, bool unique,
                               gtl::MutableArraySlice<int64> batch) const {
  SampleBatchGetExpectedCount(
      rnd, unique, batch, gtl::MutableArraySlice<float>(),
      gtl::ArraySlice<int64>(), gtl::MutableArraySlice<float>());
}

void RangeSampler::SampleBatchGetExpectedCount(
    random::SimplePhilox* rnd, bool unique, gtl::MutableArraySlice<int64> batch,
    gtl::MutableArraySlice<float> batch_expected_count,
    gtl::ArraySlice<int64> extras,
    gtl::MutableArraySlice<float> extras_expected_count) const {
  SampleBatchGetExpectedCountAvoid(rnd, unique, batch, batch_expected_count,
                                   extras, extras_expected_count,
                                   gtl::ArraySlice<int64>());
}

namespace {

// Approximates the expected count of a value in the output of SampleBatch.
//
// If unique=false, then this is (Probability(value) * batch_size)
//
// We use batch_size and num_tries, where num_tries is the observed number of
// tries it took to get batch_size unique values.
//
// Assuming (falsely) that the nubmer of tries to get a batch of batch_size
// distinct values is _always_ num_tries, the probability that the value
// is in a batch is (1 - (1-p)^num_tries)
static float ExpectedCountHelper(float p, int batch_size, int num_tries) {
  if (num_tries == batch_size) {
    // This shortcut will always be taken if unique=false
    return p * batch_size;
  }
  // numerically stable version of (1 - (1-p)^num_tries)
  return -expm1(num_tries * log1p(-p));
}

}  // namespace

void RangeSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, MutableArraySlice<int64> batch,
    MutableArraySlice<float> batch_expected_count, ArraySlice<int64> extras,
    MutableArraySlice<float> extras_expected_count,
    ArraySlice<int64> avoided_values) const {
  const int batch_size = batch.size();
  int num_tries;

  if (unique) {
    CHECK_LE(batch_size + avoided_values.size(), range_);
    std::unordered_set<int64> used(batch_size);
    used.insert(avoided_values.begin(), avoided_values.end());
    int num_picked = 0;
    num_tries = 0;
    while (num_picked < batch_size) {
      num_tries++;
      CHECK_LT(num_tries, kint32max);
      int64 value = Sample(rnd);
      if (gtl::InsertIfNotPresent(&used, value)) {
        batch[num_picked++] = value;
      }
    }
  } else {
    CHECK_EQ(avoided_values.size(), 0)
        << "avoided_values only supported with unique=true";
    for (int i = 0; i < batch_size; i++) {
      batch[i] = Sample(rnd);
    }
    num_tries = batch_size;
  }
  // Compute the expected counts of the batch and the extra values
  if (batch_expected_count.size() > 0) {
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

AllSampler::AllSampler(int64 range)
    : RangeSampler(range), inv_range_(1.0 / range) {}

void AllSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, MutableArraySlice<int64> batch,
    MutableArraySlice<float> batch_expected_count, ArraySlice<int64> extras,
    MutableArraySlice<float> extras_expected_count,
    ArraySlice<int64> avoided_values) const {
  const int batch_size = batch.size();
  CHECK_EQ(range_, batch_size);
  for (int i = 0; i < batch_size; i++) {
    batch[i] = i;
  }
  if (batch_expected_count.size() > 0) {
    CHECK_EQ(batch_size, batch_expected_count.size());
    for (int i = 0; i < batch_size; i++) {
      batch_expected_count[i] = 1;
    }
  }
  CHECK_EQ(0, avoided_values.size());
  CHECK_EQ(extras.size(), extras_expected_count.size());
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] = 1;
  }
}

UniformSampler::UniformSampler(int64 range)
    : RangeSampler(range), inv_range_(1.0 / range) {}

int64 UniformSampler::Sample(random::SimplePhilox* rnd) const {
  return rnd->Uniform64(range_);
}

float UniformSampler::Probability(int64 value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64 range)
    : RangeSampler(range), log_range_(log(range + 1)) {}

int64 LogUniformSampler::Sample(random::SimplePhilox* rnd) const {
  const int64 value =
      static_cast<int64>(exp(rnd->RandDouble() * log_range_)) - 1;
  CHECK_GE(value, 0);
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range_;
}

float LogUniformSampler::Probability(int64 value) const {
  // value is returned iff the call to UniformDouble(log_range_) in the
  // Sample() function returns a value between log(value + 1)
  // and log(value + 2).   The probability of this is:
  // (log(value + 2) - log(value + 1)) / log_range
  // To avoid two calls to log(), we compute this as follows:
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

ThreadUnsafeUnigramSampler::ThreadUnsafeUnigramSampler(int64 range)
    : RangeSampler(range), picker_(range) {
  CHECK_LT(range, kint32max);
}

int64 ThreadUnsafeUnigramSampler::Sample(random::SimplePhilox* rnd) const {
  return picker_.Pick(rnd);
}

float ThreadUnsafeUnigramSampler::Probability(int64 value) const {
  return static_cast<float>(picker_.get_weight(value)) / picker_.total_weight();
}

void ThreadUnsafeUnigramSampler::Update(ArraySlice<int64> values) {
  int num_updates = std::min(static_cast<int>(values.size()),
                             kint32max - picker_.total_weight());
  for (int i = 0; i < num_updates; i++) {
    const int64 value = values[i];
    picker_.set_weight(value, picker_.get_weight(value) + 1);
  }
}

// Thread-safe unigram sampler
UnigramSampler::UnigramSampler(int64 range)
    : RangeSampler(range), unsafe_sampler_(range) {
  CHECK_LT(range, kint32max);
}

int64 UnigramSampler::Sample(random::SimplePhilox* rnd) const {
  mutex_lock lock(mu_);  // could use reader lock
  return unsafe_sampler_.Sample(rnd);
}

float UnigramSampler::Probability(int64 value) const {
  mutex_lock lock(mu_);  // could use reader lock
  return unsafe_sampler_.Probability(value);
}

// Overriding at a high level results in far fewer lock aquisitions.
void UnigramSampler::SampleBatchGetExpectedCountAvoid(
    random::SimplePhilox* rnd, bool unique, MutableArraySlice<int64> batch,
    MutableArraySlice<float> batch_expected_count, ArraySlice<int64> extras,
    MutableArraySlice<float> extras_expected_count,
    ArraySlice<int64> avoided_values) const {
  mutex_lock lock(mu_);  // could use reader lock
  unsafe_sampler_.SampleBatchGetExpectedCountAvoid(
      rnd, unique, batch, batch_expected_count, extras, extras_expected_count,
      avoided_values);
}

void UnigramSampler::Update(ArraySlice<int64> values) {
  mutex_lock lock(mu_);
  unsafe_sampler_.Update(values);
}

FixedUnigramSampler::FixedUnigramSampler(Env* env, int64 range,
                                         const string& vocab_file,
                                         float distortion,
                                         int32 num_reserved_ids,
                                         int32 num_shards, int32 shard)
    : RangeSampler(range),
      total_weight_(0.0),
      num_shards_(num_shards),
      shard_(shard) {
  FillReservedIds(num_reserved_ids);
  // TODO(vanhoucke): make this non-crashing.
  TF_CHECK_OK(LoadFromFile(env, vocab_file, distortion));
  CHECK_EQ(range, weights_.size());
  dist_sampler_.reset(new random::DistributionSampler(weights_));
}

FixedUnigramSampler::FixedUnigramSampler(int64 range,
                                         const std::vector<float>& unigrams,
                                         float distortion,
                                         int32 num_reserved_ids,
                                         int32 num_shards, int32 shard)
    : RangeSampler(range),
      total_weight_(0.0),
      num_shards_(num_shards),
      shard_(shard) {
  FillReservedIds(num_reserved_ids);
  LoadFromUnigrams(unigrams, distortion);
  // TODO(vanhoucke): make this non-crashing.
  CHECK_EQ(range, weights_.size());
  dist_sampler_.reset(new random::DistributionSampler(weights_));
}

float FixedUnigramSampler::Probability(int64 value) const {
  return weights_.at(value) / total_weight_;
}

int64 FixedUnigramSampler::Sample(random::SimplePhilox* rnd) const {
  return dist_sampler_->Sample(rnd);
}

void FixedUnigramSampler::FillReservedIds(int32 num_reserved_ids) {
  for (int32 word_id = 0; word_id < num_reserved_ids; ++word_id) {
    if (word_id % num_shards_ == shard_) weights_.push_back(0.0);
  }
}

Status FixedUnigramSampler::LoadFromFile(Env* env, const string& vocab_file,
                                         float distortion) {
  RandomAccessFile* file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(vocab_file, &file));
  io::InputBuffer in(file, 262144 /*bytes*/);
  string line;
  int32 word_id = weights_.size();
  while (in.ReadLine(&line).ok()) {
    // The vocabulary file should be in csv like format, with the last
    // field the weight associated with the word.
    std::vector<string> cols = str_util::Split(line, ',');
    if (cols.size() == 0) continue;
    // Skip entries that do not belong to this shard.
    if (word_id % num_shards_ == shard_) {
      float w = 0.0;
      if (!strings::safe_strtof(cols.at(cols.size() - 1).c_str(), &w)) {
        return errors::InvalidArgument("Wrong vocabulary format at line: ",
                                       line);
      }
      w = pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
  return Status::OK();
}

void FixedUnigramSampler::LoadFromUnigrams(const std::vector<float>& unigrams,
                                           float distortion) {
  int32 word_id = weights_.size();
  for (float w : unigrams) {
    // Skip entries that do not belong to this shard.
    if (word_id % num_shards_ == shard_) {
      w = pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
}

}  // namespace tensorflow
