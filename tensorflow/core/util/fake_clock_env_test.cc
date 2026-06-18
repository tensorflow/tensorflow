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

#include "tensorflow/core/util/fake_clock_env.h"

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace {

class FakeClockEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fake_clock_env_ = std::make_unique<FakeClockEnv>(Env::Default());
  }

  void TearDown() override { fake_clock_env_.reset(); }

  std::unique_ptr<FakeClockEnv> fake_clock_env_;
};

TEST_F(FakeClockEnvTest, TimeInitializedToZero) {
  EXPECT_EQ(0, fake_clock_env_->NowMicros());
}

TEST_F(FakeClockEnvTest, AdvanceTimeByMicroseconds) {
  int current_time = fake_clock_env_->NowMicros();

  // Advance current time and fake clock by equal duration.
  int64_t duration = 100;
  current_time += duration;
  fake_clock_env_->AdvanceByMicroseconds(duration);
  EXPECT_EQ(current_time, fake_clock_env_->NowMicros());

  // Multiple advancements of current time and fake clock.
  for (int i = 0; i < 5; ++i) {
    fake_clock_env_->AdvanceByMicroseconds(100);
    current_time += 100;
  }
  EXPECT_EQ(current_time, fake_clock_env_->NowMicros());

  // Advance current time and fake clock by unequal durations.
  current_time += duration;
  duration = 200;
  fake_clock_env_->AdvanceByMicroseconds(duration);
  EXPECT_NE(current_time, fake_clock_env_->NowMicros());
}

}  // namespace
}  // namespace tensorflow
