/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/tfdataz_metrics.h"

#include <memory>
#include <utility>

#include "absl/time/time.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/fake_clock_env.h"

namespace tensorflow {
namespace data {
namespace {

static int64_t k1MinutesInMicros = absl::ToInt64Microseconds(absl::Minutes(1));
static int64_t k2MinutesInMicros = absl::ToInt64Microseconds(absl::Minutes(2));
static int64_t k5MinutesInMicros = absl::ToInt64Microseconds(absl::Minutes(5));
static int64_t k59MinutesInMicros =
    absl::ToInt64Microseconds(absl::Minutes(59));
static int64_t k60MinutesInMicros =
    absl::ToInt64Microseconds(absl::Minutes(60));
static int64_t k61MinutesInMicros =
    absl::ToInt64Microseconds(absl::Minutes(61));

class TfDatazMetricsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    env_ = std::make_unique<FakeClockEnv>(Env::Default());
    tfdataz_metrics_ = std::make_unique<TfDatazMetricsCollector>(
        *env_, iterator_.get(), /*model=*/nullptr);
  }

  void TearDown() override {
    env_.reset();
    tfdataz_metrics_.reset();
  }

  std::unique_ptr<DatasetBaseIterator> iterator_;
  std::unique_ptr<FakeClockEnv> env_;
  std::unique_ptr<TfDatazMetricsCollector> tfdataz_metrics_;
};

TEST_F(TfDatazMetricsTest, RecordGetNextLatency) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  tfdataz_metrics_->RecordGetNextLatency(2);
  tfdataz_metrics_->RecordGetNextLatency(3);

  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastOneMinute()),
                  2.0);
}

TEST_F(TfDatazMetricsTest, GetAverageLatencyForLastOneMinute) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  env_->AdvanceByMicroseconds(k2MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(2);
  tfdataz_metrics_->RecordGetNextLatency(3);

  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastOneMinute()),
                  2.5);
}

TEST_F(TfDatazMetricsTest, GetAverageLatencyForLastFiveMinutes) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  env_->AdvanceByMicroseconds(k5MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(4);
  tfdataz_metrics_->RecordGetNextLatency(5);
  tfdataz_metrics_->RecordGetNextLatency(6);

  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastFiveMinutes()),
                  5.0);
}

TEST_F(TfDatazMetricsTest,
       GetAverageLatencyForLastSixtyMinutesWithAdvanceBySixtyMinutes) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  env_->AdvanceByMicroseconds(k60MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(4);
  tfdataz_metrics_->RecordGetNextLatency(5);
  tfdataz_metrics_->RecordGetNextLatency(6);

  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  5.0);
}

TEST_F(TfDatazMetricsTest,
       GetAverageLatencyForLastSixtyMinutesWithAdvanceByFiftyNineMinutes) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  env_->AdvanceByMicroseconds(k59MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(4);
  tfdataz_metrics_->RecordGetNextLatency(5);
  tfdataz_metrics_->RecordGetNextLatency(6);

  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  4.0);
}

TEST_F(TfDatazMetricsTest,
       GetAverageLatencyForLastSixtyMinutesWithAdvanceBySixtyOneMinutes) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  env_->AdvanceByMicroseconds(k61MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(2);
  tfdataz_metrics_->RecordGetNextLatency(3);
  tfdataz_metrics_->RecordGetNextLatency(4);

  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  3.0);
}

TEST_F(TfDatazMetricsTest, GetMultipleAverageLatencies) {
  tfdataz_metrics_->RecordGetNextLatency(1);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastOneMinute()),
                  1.0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastFiveMinutes()),
                  1.0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  1.0);

  env_->AdvanceByMicroseconds(k1MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(2);
  tfdataz_metrics_->RecordGetNextLatency(3);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastOneMinute()),
                  2.5);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastFiveMinutes()),
                  2.0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  2.0);

  env_->AdvanceByMicroseconds(k60MinutesInMicros);
  tfdataz_metrics_->RecordGetNextLatency(4);
  tfdataz_metrics_->RecordGetNextLatency(5);
  tfdataz_metrics_->RecordGetNextLatency(6);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastOneMinute()),
                  5.0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastFiveMinutes()),
                  5.0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  5.0);
}

TEST_F(TfDatazMetricsTest, GetAverageLatencyWithZeroGetNextCalls) {
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastOneMinute()),
                  0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastFiveMinutes()),
                  0);
  EXPECT_FLOAT_EQ(absl::ToDoubleMicroseconds(
                      tfdataz_metrics_->GetAverageLatencyForLastSixtyMinutes()),
                  0);
}

class ScopedTfDataMetricsRegistration {
 public:
  explicit ScopedTfDataMetricsRegistration(
      std::shared_ptr<TfDatazMetricsCollector> collector)
      : collector_(std::move(collector)) {
    TfDatazMetricsRegistry::Register(collector_);
  }

  ~ScopedTfDataMetricsRegistration() {
    TfDatazMetricsRegistry::Deregister(collector_);
  }

  void Deregister() { TfDatazMetricsRegistry::Deregister(collector_); }

 private:
  std::shared_ptr<TfDatazMetricsCollector> collector_;
};

TEST(TfDatazMetricsRegistryTest, Register) {
  std::unique_ptr<DatasetBaseIterator> iterator;
  auto collector_one = std::make_shared<TfDatazMetricsCollector>(
      *Env::Default(), iterator.get(), /*model=*/nullptr);
  auto collector_two = std::make_shared<TfDatazMetricsCollector>(
      *Env::Default(), iterator.get(), /*model=*/nullptr);

  ScopedTfDataMetricsRegistration scoped_registration_one(collector_one);
  ScopedTfDataMetricsRegistration scoped_registration_two(collector_two);

  EXPECT_EQ(TfDatazMetricsRegistry::GetIteratorMetricCollectors().size(), 2);
}

TEST(TfDatazMetricsRegistryTest, Deregister) {
  std::unique_ptr<DatasetBaseIterator> iterator;
  auto collector_one = std::make_shared<TfDatazMetricsCollector>(
      *Env::Default(), iterator.get(), /*model=*/nullptr);
  auto collector_two = std::make_shared<TfDatazMetricsCollector>(
      *Env::Default(), iterator.get(), /*model=*/nullptr);
  auto collector_three = std::make_shared<TfDatazMetricsCollector>(
      *Env::Default(), iterator.get(), /*model=*/nullptr);
  ScopedTfDataMetricsRegistration scoped_registration_one(collector_one);
  ScopedTfDataMetricsRegistration scoped_registration_two(collector_two);
  ScopedTfDataMetricsRegistration scoped_registration_three(collector_three);
  EXPECT_EQ(TfDatazMetricsRegistry::GetIteratorMetricCollectors().size(), 3);

  scoped_registration_one.Deregister();
  EXPECT_EQ(TfDatazMetricsRegistry::GetIteratorMetricCollectors().size(), 2);

  scoped_registration_two.Deregister();
  scoped_registration_three.Deregister();
  EXPECT_EQ(TfDatazMetricsRegistry::GetIteratorMetricCollectors().size(), 0);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
