/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/rng_state_lib.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/numeric/int128.h"

namespace xla::cpu {
namespace {

TEST(RngStateTest, DeltaConstructorSetsDeltaCorrectly) {
  const int64_t test_delta = 12345;
  RngState rng(test_delta);
  EXPECT_EQ(rng.delta(), test_delta);
}

TEST(RngStateTest, ConfirmStateIsIncreasedByDelta) {
  const int64_t test_delta = 12345;
  RngState rng(test_delta);
  absl::int128 initial_data;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&initial_data));

  absl::int128 expected_data = initial_data + test_delta;

  absl::int128 data;
  rng.GetAndUpdateState(reinterpret_cast<uint64_t*>(&data));
  EXPECT_EQ(data, expected_data);
}

}  // namespace
}  // namespace xla::cpu
