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

#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {
namespace {

// Test that setting the TensorFloat-32 global variable to false causes
// TensorFloat-32 not to be used, even when the operand precision is set to the
// default.
// TODO(b/280130359): Have XLA ignore the TensorFloat-32 global variable
class TensorFloat32GlobalVarTest : public HloTestBase {
 protected:
  TensorFloat32GlobalVarTest() {
    // The error tolerances are small enough so that the use of TF32 will cause
    // the error to be greater than the tolerances.
    error_spec_ = ErrorSpec{1e-4, 1e-4};
  }
};

TEST_F(TensorFloat32GlobalVarTest, Dot) {
  tsl::enable_tensor_float_32_execution(false);
  const char* hlo_text = R"(
HloModule TestModule

ENTRY %dot_computation (x: f32[1024,1024], source: f32[1024,1024]) -> f32[1024,1024] {
  %x = f32[1024,1024] parameter(0)
  %y = f32[1024,1024] parameter(1)
  ROOT %result = f32[1024,1024] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={default, default}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, error_spec_));
}

TEST_F(TensorFloat32GlobalVarTest, Convolution) {
  tsl::enable_tensor_float_32_execution(false);
  const char* hlo_text = R"(
HloModule TestModule

ENTRY %conv_computation (x: f32[16,40,40,64], source: f32[3,3,64,64]) -> f32[16,40,40,64] {
  %x = f32[16,40,40,64] parameter(0)
  %y = f32[3,3,64,64] parameter(1)
  ROOT %result = f32[16,40,40,64] convolution(x, y), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, operand_precision={default, default}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, error_spec_));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
