/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/batching/util/periodic_function.h"

#include <memory>
#include <string>

#include "tensorflow/contrib/batching/test_util/fake_clock_env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {

namespace internal {

class PeriodicFunctionTestAccess {
 public:
  explicit PeriodicFunctionTestAccess(PeriodicFunction* periodic_function)
      : periodic_function_(periodic_function) {}

  void NotifyStop() { periodic_function_->NotifyStop(); }

 private:
  PeriodicFunction* const periodic_function_;
};

}  // namespace internal

namespace {

using test_util::FakeClockEnv;

void StopPeriodicFunction(PeriodicFunction* periodic_function,
                          FakeClockEnv* fake_clock_env,
                          const uint64 pf_interval_micros) {
  fake_clock_env->BlockUntilThreadsAsleep(1);
  internal::PeriodicFunctionTestAccess(periodic_function).NotifyStop();
  fake_clock_env->AdvanceByMicroseconds(pf_interval_micros);
}

TEST(PeriodicFunctionTest, ObeyInterval) {
  const int64 kPeriodMicros = 2;
  const int kCalls = 10;

  int actual_calls = 0;
  {
    FakeClockEnv fake_clock_env(Env::Default());
    PeriodicFunction::Options options;
    options.env = &fake_clock_env;
    PeriodicFunction periodic_function([&actual_calls]() { ++actual_calls; },
                                       kPeriodMicros, options);

    for (int i = 0; i < kCalls; ++i) {
      fake_clock_env.BlockUntilThreadsAsleep(1);
      fake_clock_env.AdvanceByMicroseconds(kPeriodMicros);
    }
    StopPeriodicFunction(&periodic_function, &fake_clock_env, kPeriodMicros);
  }

  // The function gets called kCalls+1 times: once at time 0, once at time
  // kPeriodMicros, once at time kPeriodMicros*2, up to once at time
  // kPeriodMicros*kCalls.
  ASSERT_EQ(actual_calls, kCalls + 1);
}

TEST(PeriodicFunctionTest, ObeyStartupDelay) {
  const int64 kDelayMicros = 10;
  const int64 kPeriodMicros = kDelayMicros / 10;

  int actual_calls = 0;
  {
    PeriodicFunction::Options options;
    options.startup_delay_micros = kDelayMicros;
    FakeClockEnv fake_clock_env(Env::Default());
    options.env = &fake_clock_env;
    PeriodicFunction periodic_function([&actual_calls]() { ++actual_calls; },
                                       kPeriodMicros, options);

    // Wait for the thread to start up.
    fake_clock_env.BlockUntilThreadsAsleep(1);
    // Function shouldn't have been called yet.
    EXPECT_EQ(0, actual_calls);
    // Give enough time for startup delay to expire.
    fake_clock_env.AdvanceByMicroseconds(kDelayMicros);
    StopPeriodicFunction(&periodic_function, &fake_clock_env, kDelayMicros);
  }

  // Function should have been called at least once.
  EXPECT_EQ(1, actual_calls);
}

// Test for race in calculating the first time the callback should fire.
TEST(PeriodicFunctionTest, StartupDelayRace) {
  const int64 kDelayMicros = 10;
  const int64 kPeriodMicros = kDelayMicros / 10;

  mutex mu;
  int counter = 0;
  std::unique_ptr<Notification> listener(new Notification);

  FakeClockEnv fake_clock_env(Env::Default());
  PeriodicFunction::Options options;
  options.env = &fake_clock_env;
  options.startup_delay_micros = kDelayMicros;
  PeriodicFunction periodic_function(
      [&mu, &counter, &listener]() {
        mutex_lock l(mu);
        counter++;
        listener->Notify();
      },
      kPeriodMicros, options);

  fake_clock_env.BlockUntilThreadsAsleep(1);
  fake_clock_env.AdvanceByMicroseconds(kDelayMicros);
  listener->WaitForNotification();
  {
    mutex_lock l(mu);
    EXPECT_EQ(1, counter);
    // A notification can only be notified once.
    listener.reset(new Notification);
  }
  fake_clock_env.BlockUntilThreadsAsleep(1);
  fake_clock_env.AdvanceByMicroseconds(kPeriodMicros);
  listener->WaitForNotification();
  {
    mutex_lock l(mu);
    EXPECT_EQ(2, counter);
  }
  StopPeriodicFunction(&periodic_function, &fake_clock_env, kPeriodMicros);
}

// If this test hangs forever, its probably a deadlock caused by setting the
// PeriodicFunction's interval to 0ms.
TEST(PeriodicFunctionTest, MinInterval) {
  PeriodicFunction periodic_function(
      []() { Env::Default()->SleepForMicroseconds(20 * 1000); }, 0);
}

class PeriodicFunctionWithFakeClockEnvTest : public ::testing::Test {
 protected:
  const int64 kPeriodMicros = 50;
  PeriodicFunctionWithFakeClockEnvTest()
      : fake_clock_env_(Env::Default()),
        counter_(0),
        pf_(
            [this]() {
              mutex_lock l(counter_mu_);
              ++counter_;
            },
            kPeriodMicros, GetPeriodicFunctionOptions()) {}

  PeriodicFunction::Options GetPeriodicFunctionOptions() {
    PeriodicFunction::Options options;
    options.thread_name_prefix = "ignore";
    options.env = &fake_clock_env_;
    return options;
  }

  void SetUp() override {
    // Note: counter_ gets initially incremented at time 0.
    ASSERT_TRUE(AwaitCount(1));
  }

  void TearDown() override {
    StopPeriodicFunction(&pf_, &fake_clock_env_, kPeriodMicros);
  }

  // The FakeClockEnv tests below advance simulated time and then expect the
  // PeriodicFunction thread to run its function. This method helps the tests
  // wait for the thread to execute, and then check the count matches the
  // expectation.
  bool AwaitCount(int expected_counter) {
    fake_clock_env_.BlockUntilThreadsAsleep(1);
    {
      mutex_lock lock(counter_mu_);
      return counter_ == expected_counter;
    }
  }

  FakeClockEnv fake_clock_env_;
  mutex counter_mu_;
  int counter_;
  PeriodicFunction pf_;
};

TEST_F(PeriodicFunctionWithFakeClockEnvTest, FasterThanRealTime) {
  fake_clock_env_.AdvanceByMicroseconds(kPeriodMicros / 2);
  for (int i = 2; i < 7; ++i) {
    fake_clock_env_.AdvanceByMicroseconds(
        kPeriodMicros);  // advance past a tick
    EXPECT_TRUE(AwaitCount(i));
  }
}

TEST_F(PeriodicFunctionWithFakeClockEnvTest, SlowerThanRealTime) {
  Env::Default()->SleepForMicroseconds(
      125 * 1000);  // wait for any unexpected breakage
  EXPECT_TRUE(AwaitCount(1));
}

TEST(PeriodicFunctionDeathTest, BadInterval) {
  EXPECT_DEBUG_DEATH(PeriodicFunction periodic_function([]() {}, -1),
                     ".* should be >= 0");

  EXPECT_DEBUG_DEATH(PeriodicFunction periodic_function(
                         []() {}, -1, PeriodicFunction::Options()),
                     ".* should be >= 0");
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
