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

#include "tensorflow/compiler/xla/service/gpu/hlo_op_profiler.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using HloOpProfilerTest = HloTestBase;

TEST_F(HloOpProfilerTest, BasicMeasurementsAreCorrect) {
  HloOpProfiler profiler(test_runner_);
  // f32 add is too fast to be measurable here.
  EXPECT_FALSE(
      profiler.MeasureClockCyclesPerOp(HloOpcode::kAdd, true, F32, 1).ok());
  // f64 divide is somewhat slow.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kDivide, true, F64, 1)
                .value()
                .clock_cycles(),
            500);
  // c128 sqrt is slow.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kSqrt, false, C128, 1)
                .value()
                .clock_cycles(),
            5000);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
