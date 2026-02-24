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
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

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
              absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT));
}

TEST_F(HloOpProfilerTest, AllSupportedCombinationsAreMeasurable) {
  absl::flat_hash_set<HloOpcode> FloatTypes = {
      // go/keep-sorted start
      HloOpcode::kAtan2,
      HloOpcode::kCbrt,
      HloOpcode::kCeil,
      HloOpcode::kCos,
      HloOpcode::kErf,
      HloOpcode::kExp,
      HloOpcode::kExpm1,
      HloOpcode::kFloor,
      HloOpcode::kImag,
      HloOpcode::kIsFinite,
      HloOpcode::kLog,
      HloOpcode::kLog1p,
      HloOpcode::kLogistic,
      HloOpcode::kReal,
      HloOpcode::kRoundNearestAfz,
      HloOpcode::kRoundNearestEven,
      HloOpcode::kRsqrt,
      HloOpcode::kSin,
      HloOpcode::kSqrt,
      HloOpcode::kTan,
      HloOpcode::kTanh
      // go/keep-sorted end
  };
  absl::flat_hash_set<HloOpcode> MeasurebleInFloat = {
      // go/keep-sorted start
      HloOpcode::kAdd,
      HloOpcode::kMultiply,
      HloOpcode::kSubtract,
      // go/keep-sorted end
  };

  FloatTypes.insert(MeasurebleInFloat.begin(), MeasurebleInFloat.end());
  HloOpProfiler profiler(test_runner_as_hlo_runner());
  for (const HloOpcode op : HloOpProfiler::AllSupportedOps()) {
    if (!HloOpProfiler::TooFastToMeasure().count(op) &&
        !HloOpProfiler::Unsupported().count(op)) {
      auto Type = FloatTypes.count(op) ? F32 : S32;
      TF_EXPECT_OK(profiler.MeasureClockCyclesPerOp(op, Type));
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
