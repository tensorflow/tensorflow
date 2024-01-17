/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// Testing configuration of denormal state.
#include "tsl/platform/denormal.h"

#include <cstring>
#include <limits>

#include "tsl/platform/test.h"

namespace tsl {
namespace port {

TEST(DenormalStateTest, ConstructorAndAccessorsWork) {
  const bool flush_to_zero[] = {true, true, false, false};
  const bool denormals_are_zero[] = {true, false, true, false};
  for (int i = 0; i < 4; ++i) {
    const DenormalState state =
        DenormalState(flush_to_zero[i], denormals_are_zero[i]);
    EXPECT_EQ(state.flush_to_zero(), flush_to_zero[i]);
    EXPECT_EQ(state.denormals_are_zero(), denormals_are_zero[i]);
  }
}

// Convert a 32-bit float to its binary representation.
uint32_t bits(float x) {
  uint32_t out;
  memcpy(&out, &x, sizeof(float));
  return out;
}

void CheckDenormalHandling(const DenormalState& state) {
  // Notes:
  //  - In the following tests we need to compare binary representations because
  //    floating-point comparisons can trigger denormal flushing on SSE/ARM.
  //  - We also require the input value to be marked `volatile` to prevent the
  //    compiler from optimizing away any floating-point operations that might
  //    otherwise be expected to flush denormals.

  // The following is zero iff denormal outputs are flushed to zero.
  volatile float denormal_output = std::numeric_limits<float>::min();
  denormal_output *= 0.25f;
  if (state.flush_to_zero()) {
    EXPECT_EQ(bits(denormal_output), 0x0);
  } else {
    EXPECT_NE(bits(denormal_output), 0x0);
  }

  // The following is zero iff denormal inputs are flushed to zero.
  volatile float normal_output = std::numeric_limits<float>::denorm_min();
  normal_output *= std::numeric_limits<float>::max();
  if (state.denormals_are_zero()) {
    EXPECT_EQ(bits(normal_output), 0x0);
  } else {
    EXPECT_NE(bits(normal_output), 0x0);
  }
}

TEST(DenormalTest, GetAndSetStateWorkWithCorrectFlushing) {
  const DenormalState states[] = {
      DenormalState(/*flush_to_zero=*/true, /*denormals_are_zero=*/true),
      DenormalState(/*flush_to_zero=*/true, /*denormals_are_zero=*/false),
      DenormalState(/*flush_to_zero=*/false, /*denormals_are_zero=*/true),
      DenormalState(/*flush_to_zero=*/false, /*denormals_are_zero=*/false)};

  for (const DenormalState& state : states) {
    if (SetDenormalState(state)) {
      EXPECT_EQ(GetDenormalState(), state);
      CheckDenormalHandling(state);
    }
  }
}

TEST(ScopedRestoreFlushDenormalStateTest, RestoresState) {
  const DenormalState flush_denormals(/*flush_to_zero=*/true,
                                      /*denormals_are_zero=*/true);
  const DenormalState dont_flush_denormals(/*flush_to_zero=*/false,
                                           /*denormals_are_zero=*/false);

  // Only test if the platform supports setting the denormal state.
  const bool can_set_denormal_state = SetDenormalState(flush_denormals) &&
                                      SetDenormalState(dont_flush_denormals);
  if (can_set_denormal_state) {
    // Flush -> Don't Flush -> Flush.
    SetDenormalState(flush_denormals);
    {
      ScopedRestoreFlushDenormalState restore_state;
      SetDenormalState(dont_flush_denormals);
      EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), flush_denormals);

    // Don't Flush -> Flush -> Don't Flush.
    SetDenormalState(dont_flush_denormals);
    {
      ScopedRestoreFlushDenormalState restore_state;
      SetDenormalState(flush_denormals);
      EXPECT_EQ(GetDenormalState(), flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
  }
}

TEST(ScopedFlushDenormalTest, SetsFlushingAndRestoresState) {
  const DenormalState flush_denormals(/*flush_to_zero=*/true,
                                      /*denormals_are_zero=*/true);
  const DenormalState dont_flush_denormals(/*flush_to_zero=*/false,
                                           /*denormals_are_zero=*/false);

  // Only test if the platform supports setting the denormal state.
  const bool can_set_denormal_state = SetDenormalState(flush_denormals) &&
                                      SetDenormalState(dont_flush_denormals);
  if (can_set_denormal_state) {
    SetDenormalState(dont_flush_denormals);
    {
      ScopedFlushDenormal scoped_flush_denormal;
      EXPECT_EQ(GetDenormalState(), flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
  }
}

TEST(ScopedDontFlushDenormalTest, SetsNoFlushingAndRestoresState) {
  const DenormalState flush_denormals(/*flush_to_zero=*/true,
                                      /*denormals_are_zero=*/true);
  const DenormalState dont_flush_denormals(/*flush_to_zero=*/false,
                                           /*denormals_are_zero=*/false);

  // Only test if the platform supports setting the denormal state.
  const bool can_set_denormal_state = SetDenormalState(flush_denormals) &&
                                      SetDenormalState(dont_flush_denormals);
  if (can_set_denormal_state) {
    SetDenormalState(flush_denormals);
    {
      ScopedDontFlushDenormal scoped_dont_flush_denormal;
      EXPECT_EQ(GetDenormalState(), dont_flush_denormals);
    }
    EXPECT_EQ(GetDenormalState(), flush_denormals);
  }
}

}  // namespace port
}  // namespace tsl
