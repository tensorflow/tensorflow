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

#include "xla/service/gpu/transforms/add_tracking_suffix_to_instruction_names.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace gpu {
namespace {

class RenameInstructionsTest : public HloHardwareIndependentTestBase {};

TEST_F(RenameInstructionsTest, BasicCase) {
  absl::string_view kHlo = R"(
  HloModule m

  ENTRY main {
    param_0.315 = f32[] parameter(0)
    log.482 = f32[] log(param_0.315)
    param_1.426 = f32[] parameter(1)
    subtract.22 = f32[] subtract(log.482, param_1.426)
    ROOT exponential.15 = f32[] exponential(subtract.22)
  }
  )";
  RunAndFilecheckHloRewrite(kHlo, AddTrackingSuffixToInstructionNames(), R"(
  // CHECK: ENTRY %main {{.*}} {
  // CHECK:   %param_0.315 = f32[] parameter(0)
  // CHECK:   %log.482.0 = f32[] log(%param_0.315)
  // CHECK:   %param_1.426 = f32[] parameter(1)
  // CHECK:   %subtract.22.0 = f32[] subtract(%log.482.0, %param_1.426)
  // CHECK:   ROOT %exponential.15.0 = f32[] exponential(%subtract.22.0)
  // CHECK: })");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
