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

#include <vector>

#include "tensorflow/core/kernels/range_sampler.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using gtl::ArraySlice;
using gtl::MutableArraySlice;

class RangeSamplerTest : public ::testing::Test {
 protected:
  void CheckProbabilitiesSumToOne() {
    double sum = 0;
    for (int i = 0; i < sampler_->range(); i++) {
      sum += sampler_->Probability(i);
    }
    EXPECT_NEAR(sum, 1.0, 1e-4);
  }
  void CheckHistogram(int num_samples, float tolerance) {
    const int range = sampler_->range();
    std::vector<int> h(range);
    std::vector<int64_t> a(num_samples);
    // Using a fixed random seed to make the test deterministic.
    random::PhiloxRandom philox(123, 17);
    random::SimplePhilox rnd(&philox);
    sampler_->SampleBatch(&rnd, false, absl::MakeSpan(a));
    for (int i = 0; i < num_samples; i++) {
      int64_t val = a[i];
      ASSERT_GE(val, 0);
      ASSERT_LT(val, range);
      h[val]++;
    }
    for (int val = 0; val < range; val++) {
      EXPECT_NEAR((h[val] + 0.0) / num_samples, sampler_->Probability(val),
                  tolerance);
    }
  }
  void Update1() {
    // Add the value 3 ten times.
    std::vector<int64_t> a(10);
    for (int i = 0; i < 10; i++) {
      a[i] = 3;
    }
    sampler_->Update(a);
  }
  void Update2() {
    // Add the value n times.
    int64_t a[10];
    for (int i = 0; i < 10; i++) {
      a[i] = i;
    }
    for (int64_t i = 1; i < 10; i++) {
      sampler_->Update(ArraySlice<int64_t>(a + i, 10 - i));
    }
  }
  std::unique_ptr<RangeSampler> sampler_;
};

TEST_F(RangeSamplerTest, UniformProbabilities) {
  sampler_.reset(new UniformSampler(10));
  for (int i = 0; i < 10; i++) {
    CHECK_EQ(sampler_->Probability(i), sampler_->Probability(0));
  }
}

TEST_F(RangeSamplerTest, UniformChecksum) {
  sampler_.reset(new UniformSampler(10));
  CheckProbabilitiesSumToOne();
}

TEST_F(RangeSamplerTest, UniformHistogram) {
  sampler_.reset(new UniformSampler(10));
  CheckHistogram(1000, 0.05);
}

TEST_F(RangeSamplerTest, LogUniformProbabilities) {
  int range = 1000000;
  sampler_.reset(new LogUniformSampler(range));
  for (int i = 100; i < range; i *= 2) {
    float ratio = sampler_->Probability(i) / sampler_->Probability(i / 2);
    EXPECT_NEAR(ratio, 0.5, 0.1);
  }
}

TEST_F(RangeSamplerTest, LogUniformChecksum) {
  sampler_.reset(new LogUniformSampler(10));
  CheckProbabilitiesSumToOne();
}

TEST_F(RangeSamplerTest, LogUniformHistogram) {
  sampler_.reset(new LogUniformSampler(10));
  CheckHistogram(1000, 0.05);
}

TEST_F(RangeSamplerTest, UnigramProbabilities1) {
  sampler_.reset(new UnigramSampler(10));
  Update1();
  EXPECT_NEAR(sampler_->Probability(3), 0.55, 1e-4);
  for (int i = 0; i < 10; i++) {
    if (i != 3) {
      ASSERT_NEAR(sampler_->Probability(i), 0.05, 1e-4);
    }
  }
}
TEST_F(RangeSamplerTest, UnigramProbabilities2) {
  sampler_.reset(new UnigramSampler(10));
  Update2();
  for (int i = 0; i < 10; i++) {
    ASSERT_NEAR(sampler_->Probability(i), (i + 1) / 55.0, 1e-4);
  }
}
TEST_F(RangeSamplerTest, UnigramChecksum) {
  sampler_.reset(new UnigramSampler(10));
  Update1();
  CheckProbabilitiesSumToOne();
}

TEST_F(RangeSamplerTest, UnigramHistogram) {
  sampler_.reset(new UnigramSampler(10));
  Update1();
  CheckHistogram(1000, 0.05);
}

static const char kVocabContent[] =
    "w1,1\n"
    "w2,2\n"
    "w3,4\n"
    "w4,8\n"
    "w5,16\n"
    "w6,32\n"
    "w7,64\n"
    "w8,128\n"
    "w9,256";
TEST_F(RangeSamplerTest, FixedUnigramProbabilities) {
  Env* env = Env::Default();
  string fname = io::JoinPath(testing::TmpDir(), "vocab_file");
  TF_CHECK_OK(WriteStringToFile(env, fname, kVocabContent));
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(env, fname));
  sampler_.reset(test_sampler);
  // 1^0.8+2^0.8+4^0.8+...+256^0.8=197.05
  for (int i = 0; i < 9; i++) {
    ASSERT_NEAR(sampler_->Probability(i), pow(2, i * 0.8) / 197.05, 1e-4);
  }
}
TEST_F(RangeSamplerTest, FixedUnigramNoExistingFilename) {
  Env* env = Env::Default();
  string fname = "NoExistingFile";
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  Status s = test_sampler->SetDistributionSampler(env, fname);
  sampler_.reset(test_sampler);
  EXPECT_TRUE(errors::IsNotFound(s)) << s;
}
TEST_F(RangeSamplerTest, FixedUnigramNoMatchingRangeWeights) {
  Env* env = Env::Default();
  string fname = io::JoinPath(testing::TmpDir(), "vocab_file");
  TF_CHECK_OK(WriteStringToFile(env, fname, kVocabContent));
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(8, 0.8, 0, 1, 0);
  Status s = test_sampler->SetDistributionSampler(env, fname);
  sampler_.reset(test_sampler);
  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;
}
TEST_F(RangeSamplerTest, FixedUnigramChecksum) {
  Env* env = Env::Default();
  string fname = io::JoinPath(testing::TmpDir(), "vocab_file");
  TF_CHECK_OK(WriteStringToFile(env, fname, kVocabContent));
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(env, fname));
  sampler_.reset(test_sampler);
  CheckProbabilitiesSumToOne();
}
TEST_F(RangeSamplerTest, FixedUnigramHistogram) {
  Env* env = Env::Default();
  string fname = io::JoinPath(testing::TmpDir(), "vocab_file");
  TF_CHECK_OK(WriteStringToFile(env, fname, kVocabContent));
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(env, fname));
  sampler_.reset(test_sampler);
  CheckHistogram(1000, 0.05);
}
TEST_F(RangeSamplerTest, FixedUnigramProbabilitiesReserve1) {
  Env* env = Env::Default();
  string fname = io::JoinPath(testing::TmpDir(), "vocab_file");
  TF_CHECK_OK(WriteStringToFile(env, fname, kVocabContent));
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(10, 0.8, 1, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(env, fname));
  sampler_.reset(test_sampler);
  ASSERT_NEAR(sampler_->Probability(0), 0, 1e-4);
  // 1^0.8+2^0.8+4^0.8+...+256^0.8=197.05
  for (int i = 1; i < 10; i++) {
    ASSERT_NEAR(sampler_->Probability(i), pow(2, (i - 1) * 0.8) / 197.05, 1e-4);
  }
}
TEST_F(RangeSamplerTest, FixedUnigramProbabilitiesReserve2) {
  Env* env = Env::Default();
  string fname = io::JoinPath(testing::TmpDir(), "vocab_file");
  TF_CHECK_OK(WriteStringToFile(env, fname, kVocabContent));
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(11, 0.8, 2, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(env, fname));
  sampler_.reset(test_sampler);
  ASSERT_NEAR(sampler_->Probability(0), 0, 1e-4);
  ASSERT_NEAR(sampler_->Probability(1), 0, 1e-4);
  // 1^0.8+2^0.8+4^0.8+...+256^0.8=197.05
  for (int i = 2; i < 11; i++) {
    ASSERT_NEAR(sampler_->Probability(i), pow(2, (i - 2) * 0.8) / 197.05, 1e-4);
  }
}
TEST_F(RangeSamplerTest, FixedUnigramProbabilitiesFromVector) {
  std::vector<float> weights = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(weights));
  sampler_.reset(test_sampler);
  // 1^0.8+2^0.8+4^0.8+...+256^0.8=197.05
  for (int i = 0; i < 9; i++) {
    ASSERT_NEAR(sampler_->Probability(i), pow(2, i * 0.8) / 197.05, 1e-4);
  }
}
TEST_F(RangeSamplerTest, FixedUnigramChecksumFromVector) {
  std::vector<float> weights = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(weights));
  sampler_.reset(test_sampler);
  CheckProbabilitiesSumToOne();
}
TEST_F(RangeSamplerTest, FixedUnigramHistogramFromVector) {
  std::vector<float> weights = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(9, 0.8, 0, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(weights));
  sampler_.reset(test_sampler);
  CheckHistogram(1000, 0.05);
}
TEST_F(RangeSamplerTest, FixedUnigramProbabilitiesReserve1FromVector) {
  std::vector<float> weights = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(10, 0.8, 1, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(weights));
  sampler_.reset(test_sampler);
  ASSERT_NEAR(sampler_->Probability(0), 0, 1e-4);
  // 1^0.8+2^0.8+4^0.8+...+256^0.8=197.05
  for (int i = 1; i < 10; i++) {
    ASSERT_NEAR(sampler_->Probability(i), pow(2, (i - 1) * 0.8) / 197.05, 1e-4);
  }
}
TEST_F(RangeSamplerTest, FixedUnigramProbabilitiesReserve2FromVector) {
  std::vector<float> weights = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  FixedUnigramSampler* test_sampler = new FixedUnigramSampler(11, 0.8, 2, 1, 0);
  TF_CHECK_OK(test_sampler->SetDistributionSampler(weights));
  sampler_.reset(test_sampler);
  ASSERT_NEAR(sampler_->Probability(0), 0, 1e-4);
  ASSERT_NEAR(sampler_->Probability(1), 0, 1e-4);
  // 1^0.8+2^0.8+4^0.8+...+256^0.8=197.05
  for (int i = 2; i < 11; i++) {
    ASSERT_NEAR(sampler_->Probability(i), pow(2, (i - 2) * 0.8) / 197.05, 1e-4);
  }
}

// AllSampler cannot call Sample or Probability directly.
// We will test SampleBatchGetExpectedCount instead.
TEST_F(RangeSamplerTest, All) {
  int batch_size = 10;
  sampler_.reset(new AllSampler(10));
  std::vector<int64_t> batch(batch_size);
  std::vector<float> batch_expected(batch_size);
  std::vector<int64_t> extras(2);
  std::vector<float> extras_expected(2);
  extras[0] = 0;
  extras[1] = batch_size - 1;
  sampler_->SampleBatchGetExpectedCount(nullptr,  // no random numbers needed
                                        false, absl::MakeSpan(batch),
                                        absl::MakeSpan(batch_expected), extras,
                                        absl::MakeSpan(extras_expected));
  for (int i = 0; i < batch_size; i++) {
    EXPECT_EQ(i, batch[i]);
    EXPECT_EQ(1, batch_expected[i]);
  }
  EXPECT_EQ(1, extras_expected[0]);
  EXPECT_EQ(1, extras_expected[1]);
}

TEST_F(RangeSamplerTest, Unique) {
  // We sample num_batches batches, each without replacement.
  //
  // We check that the returned expected counts roughly agree with each other
  // and with the average observed frequencies over the set of batches.
  random::PhiloxRandom philox(123, 17);
  random::SimplePhilox rnd(&philox);
  const int range = 100;
  const int batch_size = 50;
  const int num_batches = 100;
  sampler_.reset(new LogUniformSampler(range));
  std::vector<int> histogram(range);
  std::vector<int64_t> batch(batch_size);
  std::vector<int64_t> all_values(range);
  for (int i = 0; i < range; i++) {
    all_values[i] = i;
  }
  std::vector<float> expected(range);

  // Sample one batch and get the expected counts of all values
  sampler_->SampleBatchGetExpectedCount(&rnd, true, absl::MakeSpan(batch),
                                        MutableArraySlice<float>(), all_values,
                                        absl::MakeSpan(expected));
  // Check that all elements are unique
  std::set<int64_t> s(batch.begin(), batch.end());
  CHECK_EQ(batch_size, s.size());

  for (int trial = 0; trial < num_batches; trial++) {
    std::vector<float> trial_expected(range);
    sampler_->SampleBatchGetExpectedCount(
        &rnd, true, absl::MakeSpan(batch), MutableArraySlice<float>(),
        all_values, absl::MakeSpan(trial_expected));
    for (int i = 0; i < range; i++) {
      EXPECT_NEAR(expected[i], trial_expected[i], expected[i] * 0.5);
    }
    for (int i = 0; i < batch_size; i++) {
      histogram[batch[i]]++;
    }
  }
  for (int i = 0; i < range; i++) {
    // Check that the computed expected count agrees with the average observed
    // count.
    const float average_count = static_cast<float>(histogram[i]) / num_batches;
    EXPECT_NEAR(expected[i], average_count, 0.2);
  }
}

TEST_F(RangeSamplerTest, Avoid) {
  random::PhiloxRandom philox(123, 17);
  random::SimplePhilox rnd(&philox);
  sampler_.reset(new LogUniformSampler(100));
  std::vector<int64_t> avoided(2);
  avoided[0] = 17;
  avoided[1] = 23;
  std::vector<int64_t> batch(98);

  // We expect to pick all elements of [0, 100) except the avoided two.
  sampler_->SampleBatchGetExpectedCountAvoid(
      &rnd, true, absl::MakeSpan(batch), MutableArraySlice<float>(),
      ArraySlice<int64_t>(), MutableArraySlice<float>(), avoided);

  int sum = 0;
  for (auto val : batch) {
    sum += val;
  }
  const int expected_sum = 100 * 99 / 2 - avoided[0] - avoided[1];
  EXPECT_EQ(expected_sum, sum);
}

}  // namespace

}  // namespace tensorflow
