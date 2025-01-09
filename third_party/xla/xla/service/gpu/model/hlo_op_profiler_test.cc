/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/hlo_op_profiler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

class HloOpProfilerTest : public HloTestBase {
  void SetUp() override {
#ifndef GOOGLE_CUDA
    GTEST_SKIP() << "Not built with --config=cuda";
#endif
  }
};

TEST_F(HloOpProfilerTest, BasicMeasurementsAreCorrect) {
  HloOpProfiler profiler(test_runner_as_hlo_runner());
  // f32 is fast but measurable.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kAdd, F32)
                .value()
                .clock_cycles(),
            0);
  // f64 divide is somewhat slow.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kDivide, F64)
                .value()
                .clock_cycles(),
            280);
  // c128 sqrt is slow.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kSqrt, C128)
                .value()
                .clock_cycles(),
            1000);
}

TEST_F(HloOpProfilerTest, UnsupportedCombinationsDoNotCrash) {
  HloOpProfiler profiler(test_runner_as_hlo_runner());
  EXPECT_THAT(profiler.MeasureClockCyclesPerOp(HloOpcode::kCbrt, S8),
              tsl::testing::StatusIs(tsl::error::INVALID_ARGUMENT));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
