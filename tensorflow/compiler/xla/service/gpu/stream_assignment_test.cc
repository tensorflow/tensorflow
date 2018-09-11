/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

class StreamAssignmentTest : public HloVerifiedTestBase {
 protected:
  std::unique_ptr<HloModule> CreateNewModule() {
    HloModuleConfig config;
    auto debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_disable_multi_streaming(false);
    config.set_debug_options(debug_options);
    return absl::make_unique<HloModule>("test_module", config);
  }

  // Pre-canned shapes.
  Shape f32_2x2_ = ShapeUtil::MakeShape(F32, {2, 2});
};

TEST_F(StreamAssignmentTest, SequentialMatMul) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* dot1 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, x, y));
  HloInstruction* dot2 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, dot1, z));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(dot2));

  std::unique_ptr<StreamAssignment> assignment = AssignStreams(*module);
  EXPECT_EQ(assignment->StreamNumberForHlo(*dot1),
            assignment->StreamNumberForHlo(*dot2));
}

TEST_F(StreamAssignmentTest, ConcurrentMatMul) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* dot1 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, x, y));
  HloInstruction* dot2 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, y, x));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, dot1, dot2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(add));

  std::unique_ptr<StreamAssignment> assignment = AssignStreams(*module);
  EXPECT_NE(assignment->StreamNumberForHlo(*dot1),
            assignment->StreamNumberForHlo(*dot2));
}

TEST_F(StreamAssignmentTest, LatticeMatMul) {
  //      d00      -- layer 0
  //     /   \
  //   d10   d11   -- layer 1
  //  /   \ /   \
  // d20  d21  d22 -- layer 2
  //  \   / \   /
  //   d30   d31   -- layer 3
  //     \   /
  //      d40      -- layer 4
  HloComputation::Builder builder("entry_computation");
  std::vector<HloInstruction*> params;
  params.reserve(6);
  for (int i = 0; i < 6; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, f32_2x2_, /*name=*/absl::StrFormat("param%d", i))));
  }
  HloInstruction* d00 = builder.AddInstruction(
      CreateCanonicalDot(f32_2x2_, params[2], params[3]));
  HloInstruction* d10 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, params[1], d00));
  HloInstruction* d11 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d00, params[4]));
  HloInstruction* d20 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, params[0], d10));
  HloInstruction* d21 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d10, d11));
  HloInstruction* d22 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d11, params[5]));
  HloInstruction* d30 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d20, d21));
  HloInstruction* d31 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d21, d22));
  HloInstruction* d40 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d30, d31));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(d40));

  std::unique_ptr<StreamAssignment> assignment = AssignStreams(*module);
  // The two dots on layer 1 are concurrent.
  EXPECT_NE(assignment->StreamNumberForHlo(*d10),
            assignment->StreamNumberForHlo(*d11));
  // The three dots on layer 2 are concurrent.
  EXPECT_NE(assignment->StreamNumberForHlo(*d20),
            assignment->StreamNumberForHlo(*d21));
  EXPECT_NE(assignment->StreamNumberForHlo(*d20),
            assignment->StreamNumberForHlo(*d22));
  EXPECT_NE(assignment->StreamNumberForHlo(*d21),
            assignment->StreamNumberForHlo(*d22));
  // The two dots on layer 3 are concurrent.
  EXPECT_NE(assignment->StreamNumberForHlo(*d30),
            assignment->StreamNumberForHlo(*d31));
}

}  // namespace gpu
}  // namespace xla
