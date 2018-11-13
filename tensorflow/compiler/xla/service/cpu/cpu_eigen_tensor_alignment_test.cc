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

#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features_fake.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace cpu {
namespace {

// Test that we don't call into Eigen with tensors too small to be aligned
// reliably.

class CpuEigenTensorAlignmentTest : public ::testing::Test {};

TEST_F(CpuEigenTensorAlignmentTest, EigenDotAlignment) {
  string hlo_string = R"(
HloModule DotOperation

ENTRY DotOperation {
  arg0 = f32[5,256] parameter(0)
  arg1 = f32[256,1024] parameter(1)
  ROOT dot = f32[5,1024] dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));

  HloInstruction* dot = module->entry_computation()->root_instruction();

  TargetMachineFeaturesWithFakeAlignmentLogic target_machine_with_no_alignment(
      [](int64 size) { return 1; });

  EXPECT_FALSE(
      PotentiallyImplementedAsEigenDot(*dot, target_machine_with_no_alignment));

  TargetMachineFeaturesWithFakeAlignmentLogic
      target_machine_with_full_alignment([](int64 size) {
        return TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });

  EXPECT_TRUE(PotentiallyImplementedAsEigenDot(
      *dot, target_machine_with_full_alignment));
}

TEST_F(CpuEigenTensorAlignmentTest, EigenConvAlignment) {
  string hlo_string = R"(
HloModule ConvOperation

ENTRY ConvOperation {
  arg0 = f32[1,2,1] parameter(0)
  arg1 = f32[1,1,1] parameter(1)
  ROOT conv = f32[1,2,1] convolution(arg0, arg1), window={size=1}, dim_labels=b0f_0io->b0f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));

  HloInstruction* conv = module->entry_computation()->root_instruction();

  TargetMachineFeaturesWithFakeAlignmentLogic target_machine_with_no_alignment(
      [](int64 size) { return 1; });

  EXPECT_FALSE(PotentiallyImplementedAsEigenConvolution(
      *conv, target_machine_with_no_alignment));

  TargetMachineFeaturesWithFakeAlignmentLogic
      target_machine_with_full_alignment([](int64 size) {
        return TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });

  EXPECT_TRUE(PotentiallyImplementedAsEigenConvolution(
      *conv, target_machine_with_full_alignment));
}
}  // namespace
}  // namespace cpu
}  // namespace xla
