/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_emission_utils.h"

#include <cstdint>
#include <memory>

#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using IrEmitterTest = HloTestBase;

TEST_F(IrEmitterTest, ConvWithZeroSizedKernelNotImplementedAsEigen) {
  const char* const hlo_string = R"(
HloModule ModuleWithConv

ENTRY Conv {
  input = f32[32,50,28,28]{3,2,1,0} parameter(0)
  kernel = f32[50,0,5,5]{3,2,1,0} parameter(1)
  ROOT convolution = f32[32,0,24,24]{3,2,1,0} convolution(input, kernel),
    window={size=5x5},
    dim_labels=bf01_io01->bf01
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* entry_computation = module->entry_computation();

  HloInstruction* conv_instr = entry_computation->root_instruction();
  cpu::TargetMachineFeaturesStub target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  EXPECT_FALSE(cpu::PotentiallyImplementedAsEigenConvolution(
      *conv_instr, target_machine_features));
}

}  // namespace
}  // namespace xla
