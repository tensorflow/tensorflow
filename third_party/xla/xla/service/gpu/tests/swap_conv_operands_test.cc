/* Copyright 2021 The OpenXLA Authors.

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

#include <memory>
#include <variant>

#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

// TODO(b/210165681): The tests in this file are fragile to HLO op names.

namespace xla {
namespace gpu {

namespace {

class SwapConvOperandsTest : public GpuCodegenTest {
 public:
  absl::StatusOr<se::GpuComputeCapability> GpuComputeCapability() {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::DeviceDescription> device_description,
        GetTestPlatform()->DescriptionForDevice(0));

    return device_description->gpu_compute_capability();
  }
};

// Here, we swap the operands of a convolution to avoid the performance penalty
// associated with convolutions with large padding. This tests that the operands
// are swapped in this case, and that the emitted convolution is successfully
// lowered to a cuDNN custom-call.
TEST_F(SwapConvOperandsTest, LargePadding) {
  const char* hlo_text = R"(
HloModule swap_conv

ENTRY swap_conv {
  input = f32[512,128,3,3]{3,2,1,0} parameter(0)
  filter = f32[1,30,30,512]{3,2,1,0}  parameter(1)
  convolution = f32[1,32,32,128]{3,2,1,0} convolution(input, filter), window={size=30x30 pad=29_29x29_29}, dim_labels=fb01_o01i->f01b
  ROOT tuple = (f32[1,32,32,128]{3,2,1,0}) tuple(convolution)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(se::GpuComputeCapability gpu_compute_capability,
                          GpuComputeCapability());

  if (std::get_if<se::CudaComputeCapability>(&gpu_compute_capability)
          ->IsAtLeastHopper()) {
    MatchOptimizedHloWithShapes(hlo_text,
                                R"(
// CHECK: [[cudnn_conv_1_0:%[^ ]+]] = (f32[1,32,32,128]{3,2,1,0}, u8[{{.*}}]{0}) custom-call(f32[1,30,30,512]{3,2,1,0} {{[^ ]+}}, f32[128,3,3,512]{3,2,1,0} {{[^ ]+}}), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
    )");
  } else {
    MatchOptimizedHloWithShapes(hlo_text,
                                R"(
// CHECK: [[cudnn_conv_1_0:%[^ ]+]] = (f32[1,128,32,32]{3,2,1,0}, u8[{{.*}}]{0}) custom-call(f32[1,512,30,30]{3,2,1,0} [[fusion_1_1:%[^ ]+]], f32[128,512,3,3]{3,2,1,0} [[transpose_1_2:%[^ ]+]]), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward"
    )");
  }

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// If the padding is already small, we leave the operands as-is before lowering.
TEST_F(SwapConvOperandsTest, SmallPadding) {
  const char* hlo_text = R"(
HloModule swap_conv

ENTRY swap_conv {
  filter = f32[512,128,3,3]{3,2,1,0} parameter(0)
  input = f32[1,30,30,512]{3,2,1,0}  parameter(1)
  convolution = f32[1,32,32,128]{2,1,3,0} convolution(input, filter), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=b01f_io01->b01f
  ROOT tuple = (f32[1,32,32,128]{3,2,1,0}) tuple(convolution)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(se::GpuComputeCapability gpu_compute_capability,
                          GpuComputeCapability());

  if (std::get_if<se::CudaComputeCapability>(&gpu_compute_capability)
          ->IsAtLeastHopper()) {
    MatchOptimizedHloWithShapes(hlo_text,
                                R"(
// CHECK: [[cudnn_conv_1_0:%[^ ]+]] = (f32[1,32,32,128]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call(f32[1,30,30,512]{3,2,1,0} {{[^ ]+}}, f32[128,3,3,512]{3,2,1,0} {{[^ ]+}}), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
    )");
  } else {
    MatchOptimizedHloWithShapes(hlo_text,
                                R"(
// CHECK: [[cudnn_conv_1_0:%[^ ]+]] = (f32[1,128,32,32]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call(f32[1,512,30,30]{3,2,1,0} [[fusion_1_1:%[^ ]+]], f32[128,512,3,3]{3,2,1,0} [[transpose_1_2:%[^ ]+]]), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward"
    )");
  }
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// If swapping the conv operands would result in a conv that does not lower to a
// valid cuDNN call, we do not transform the op.
TEST_F(SwapConvOperandsTest, DoesNotLower) {
  const char* hlo_text = R"(
HloModule swap_conv

ENTRY %conv3DBackpropInputV2(arg0.1: f32[3,3,3,2,3]) -> f32[2,4,3,3,2] {
  %constant.5 = f32[2,2,2,2,3]{4,3,2,1,0} constant({...})
  %arg0.1 = f32[3,3,3,2,3]{4,3,2,1,0} parameter(0), parameter_replication={false}
  %reshape.2 = f32[3,3,3,2,3]{4,3,2,1,0} reshape(f32[3,3,3,2,3]{4,3,2,1,0} %arg0.1)
  %reverse.6 = f32[3,3,3,2,3]{4,3,2,1,0} reverse(f32[3,3,3,2,3]{4,3,2,1,0} %reshape.2), dimensions={0,1,2}
  %convolution.7 = f32[2,4,3,3,2]{4,3,2,1,0} convolution(f32[2,2,2,2,3]{4,3,2,1,0} %constant.5, f32[3,3,3,2,3]{4,3,2,1,0} %reverse.6), window={size=3x3x3 pad=2_1x1_1x1_1 lhs_dilate=2x2x2}, dim_labels=b012f_012oi->b012f, metadata={op_type="Conv3DBackpropInputV2" op_name="gradients_2/Conv3DBackpropFilterV2_1_grad/Conv3DBackpropInputV2"}
  ROOT  %reshape.8 = f32[2,4,3,3,2]{4,3,2,1,0} reshape(f32[2,4,3,3,2]{4,3,2,1,0} %convolution.7)
}
)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK:   [[cudnn_conv_bw_input_2_0:%[^ ]+]] = (f32[2,2,5,3,3]{4,3,2,1,0}, u8[0]{0}) custom-call(f32[2,3,2,2,2]{4,3,2,1,0} [[constant_1:%[^ ]+]], f32[3,2,3,3,3]{4,3,2,1,0} [[transpose_2:%[^ ]+]]), window={size=3x3x3 stride=2x2x2 pad=0_0x1_1x1_1}, dim_labels=bf012_oi012->bf012, custom_call_target="__cudnn$convBackwardInput"
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
