/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdint>
#include <string>

#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace cpu {
namespace {

// Test that we don't call into Eigen with tensors too small to be aligned
// reliably.

using CpuEigenTensorAlignmentTest = HloTestBase;

TEST_F(CpuEigenTensorAlignmentTest, EigenConvAlignment) {
  std::string hlo_string = R"(
HloModule ConvOperation

ENTRY ConvOperation {
  arg0 = f32[1,2,1] parameter(0)
  arg1 = f32[1,1,1] parameter(1)
  ROOT conv = f32[1,2,1] convolution(arg0, arg1), window={size=1}, dim_labels=b0f_0io->b0f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* conv = module->entry_computation()->root_instruction();

  TargetMachineFeaturesStub target_machine_with_no_alignment(
      [](int64_t size) { return 1; });

  EXPECT_FALSE(PotentiallyImplementedAsEigenConvolution(
      *conv, target_machine_with_no_alignment));

  TargetMachineFeaturesStub target_machine_with_full_alignment(
      [](int64_t size) {
        return TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });

  EXPECT_TRUE(PotentiallyImplementedAsEigenConvolution(
      *conv, target_machine_with_full_alignment));
}
}  // namespace
}  // namespace cpu
}  // namespace xla
