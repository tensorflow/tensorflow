/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/compilation_timer.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace {

TEST(CompilationTimer, MeasuresElapsedTime) {
  uint64_t timer_result_in_milliseconds;

  {
    CompilationTimer timer;

    absl::SleepFor(absl::Milliseconds(100));

    timer_result_in_milliseconds = timer.ElapsedCyclesInMilliseconds();
  }

  ASSERT_THAT(timer_result_in_milliseconds, testing::Ne(0));
}

}  // namespace
