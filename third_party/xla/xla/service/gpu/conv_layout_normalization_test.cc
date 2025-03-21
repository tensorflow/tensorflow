/* Copyright 2022 The OpenXLA Authors.

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

#include <variant>

#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class ConvolutionLayoutNormalizationTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  bool IsRocm() {
    return std::holds_alternative<se::RocmComputeCapability>(
        backend()
            .default_stream_executor()
            ->GetDeviceDescription()
            .gpu_compute_capability());
  }
};

TEST_F(ConvolutionLayoutNormalizationTest, BackwardInput) {
  const char* hlo = R"(
HloModule TestModule

%TestComputation1 (param_0: f32[1,20,257], param_1: f32[31,257,136]) -> (f32[1,23,136], u8[0]) {
  %param_0 = f32[1,20,257]{2,1,0} parameter(0)
  %copy.3 = f32[1,20,257]{1,2,0} copy(f32[1,20,257]{2,1,0} %param_0)
  %param_1 = f32[31,257,136]{2,1,0} parameter(1)
  %copy.4 = f32[31,257,136]{0,2,1} copy(f32[31,257,136]{2,1,0} %param_1)
  %custom-call.1 = (f32[1,23,136]{1,2,0}, u8[0]{0}) custom-call(f32[1,20,257]{1,2,0} %copy.3, f32[31,257,136]{0,2,1} %copy.4), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config={"cudnn_conv_backend_config":{conv_result_scale:1}}
  %get-tuple-element.2 = f32[1,23,136]{1,2,0} get-tuple-element((f32[1,23,136]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %copy.5 = f32[1,23,136]{2,1,0} copy(f32[1,23,136]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[1,23,136]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[1,23,136]{2,1,0}, u8[0]{0}) tuple(f32[1,23,136]{2,1,0} %copy.5, u8[0]{0} %get-tuple-element.3)
}
)";

  if (!IsRocm() && GetCudaComputeCapability().IsAtLeastHopper()) {
    MatchOptimizedHlo(hlo, R"(
// CHECK: (f32[1,23,136]{2,1,0}, u8[{{[0-9]+}}]{0}) custom-call([[fusion_1_0:%[^ ]+]], [[transpose_1_1:%[^ ]+]]), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_o0i->b0f, custom_call_target="__cudnn$convBackwardInput"
    )");
  } else {
    MatchOptimizedHlo(hlo, R"(
// CHECK: (f32[1,136,23]{2,1,0}, u8[{{[0-9]+}}]{0}) custom-call([[fusion_1_0:%[^ ]+]], [[transpose_1_1:%[^ ]+]]), window={size=31 stride=2 pad=23_23}, dim_labels=bf0_oi0->bf0, custom_call_target="__cudnn$convBackwardInput"
  )");
  }
}

TEST_F(ConvolutionLayoutNormalizationTest, Forward) {
  const char* hlo = R"(
HloModule TestModule

ENTRY %TestComputation {
  %param_0 = f32[2,128,1,378]{3,2,1,0} parameter(0)
  %param_1 = f32[1,5,128,128]{1,0,2,3} parameter(1)
  ROOT %custom-call.1 = (f32[2,128,1,378]{3,2,1,0}, u8[0]{0}) custom-call(%param_0, %param_1), window={size=1x5 pad=0_0x2_2}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config={"cudnn_conv_backend_config":{conv_result_scale:1}}
}
)";

  if (!IsRocm() && GetCudaComputeCapability().IsAtLeastHopper()) {
    MatchOptimizedHlo(hlo, R"(
// CHECK: (f32[2,1,378,128]{3,2,1,0}, u8[{{[0-9]+}}]{0}) custom-call([[param_0_0:%[^ ]+]], [[bitcast_5_1:%[^ ]+]]), window={size=1x5 pad=0_0x2_2}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
    )");
  } else {
    MatchOptimizedHlo(hlo, R"(
// CHECK: (f32[2,128,1,378]{3,2,1,0}, u8[{{[0-9]+}}]{0}) custom-call([[param_0_0:%[^ ]+]], [[bitcast_5_1:%[^ ]+]]), window={size=1x5 pad=0_0x2_2}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward"
    )");
  }
}

TEST_F(ConvolutionLayoutNormalizationTest, FusedConv3D) {
  if (IsRocm()) {
    GTEST_SKIP() << "Conv3D is not supported on ROCm.";
  }
  const char* hlo = R"(
HloModule TestModule

ENTRY TestComputation {
  %p0 = f32[8,4,5,5,1] parameter(0)
  %p1 = f32[3,3,3,1,32] parameter(1)
  %conv = f32[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
  %bias = f32[32] parameter(2)
  %broadcasted_bias = f32[8,4,5,5,32] broadcast(%bias), dimensions={4}
  %add = f32[8,4,5,5,32] add(%conv, %broadcasted_bias)
  %zero = f32[] constant(0)
  %zeros = f32[8,4,5,5,32] broadcast(%zero), dimensions={}
  ROOT relu = f32[8,4,5,5,32] maximum(%zeros, %add)
}
)";

  MatchOptimizedHlo(hlo, R"(
// CHECK: (f32[8,32,4,5,5]{4,3,2,1,0}, u8[0]{0}) custom-call([[bitcast_8_0:%[^ ]+]], [[fusion_1:%[^ ]+]], [[bias_2:%[^ ]+]]), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=bf012_oi012->bf012, custom_call_target="__cudnn$convBiasActivationForward"
  )");
}

TEST_F(ConvolutionLayoutNormalizationTest, GraphConvF8) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::kHopper)) {
    GTEST_SKIP() << "FP8 convolutions require Hopper or newer architecture.";
  }
  const char* hlo = R"(
    HloModule Test

ENTRY %Test (input.1: f8e4m3fn[2,1,378,128], filter.1: f8e4m3fn[1,128,128,5], input_scale.1: f32[], filter_scale.1: f32[], z_scale.1: f32[]) -> (f8e4m3fn[2,1,378,128], f32[], u8[0]{0}) {
  %input.1 = f8e4m3fn[2,1,378,128]{3,2,1,0} parameter(0)
  %filter.1 = f8e4m3fn[128,1,5,128]{1,0,2,3} parameter(1)
  %input_scale.1 = f32[] parameter(2)
  %filter_scale.1 = f32[] parameter(3)
  %z_scale.1 = f32[] parameter(4)
  ROOT   %cudnn-conv.3.0 = (f8e4m3fn[2,1,378,128]{3,2,1,0}, f32[], u8[0]{0}) custom-call(%input.1, %filter.1, %input_scale.1, %filter_scale.1, %z_scale.1), window={size=1x5 pad=0_0x2_2}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph", backend_config={"cudnn_conv_backend_config":{"conv_result_scale":1,"serialized_graph":"28:[f32]conv();30:[f32]scale(28);32:[f32]scale(30);16:[f8e4m3fn]scale(32);25:[f32]amax(32);"}}
    })";

  MatchOptimizedHlo(hlo, R"(
// CHECK: (f8e4m3fn[2,1,378,128]{3,2,1,0}, f32[], u8[{{[0-9]+}}]{0}) custom-call([[INPUT:%[^ ]+]], [[FILTER:%[^ ]+]], [[INPUT_SCALE:%[^ ]+]], [[FILTER_SCALE:%[^ ]+]], [[Z_SCALE:%[^ ]+]]), window={size=1x5 pad=0_0x2_2}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
