/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"

#include "tensorflow/compiler/xla/service/cpu/target_machine_features_fake.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

namespace xla {
namespace {

TEST(IrEmitterTest, ConvWithZeroSizedKernelNotImplementedAsEigen) {
  const char* const hlo_string = R"(
HloModule ModuleWithConv

ENTRY Conv {
  input = f32[32,50,28,28]{3,2,1,0} parameter(0)
  kernel = f32[0,32,5,5]{3,2,1,0} parameter(1)
  ROOT convolution = f32[64,50,24,24]{3,2,1,0} convolution(input, kernel),
    window={size=5x5},
    dim_labels=b01f_01io->b01f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(hlo_string));

  HloComputation* entry_computation = module->entry_computation();

  HloInstruction* conv_instr = entry_computation->root_instruction();
  cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features(
      [](int64 shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  EXPECT_FALSE(cpu::PotentiallyImplementedAsEigenConvolution(
      *conv_instr, target_machine_features));
}

}  // namespace
}  // namespace xla
