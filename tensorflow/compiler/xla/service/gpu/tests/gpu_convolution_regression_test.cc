/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class GpuConvolutionRegressionTest : public HloTestBase {
 public:
  // RunHloPasses goes through convolution autotuning, which performs
  // correctness cross-checking.
  void CheckForHloText(absl::string_view hlo_string) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    (void)backend().compiler()->RunHloPasses(
        ParseAndReturnVerifiedModule(hlo_string, config).value(),
        backend().default_stream_executor(), backend().memory_allocator());
  }
};

TEST_F(GpuConvolutionRegressionTest, Computation1) {
  CheckForHloText(R"(
HloModule TestModule

%TestComputation1 (param_0: f32[1,20,257], param_1: f32[31,257,136]) -> (f32[1,23,136], u8[0]) {
  %param_0 = f32[1,20,257]{2,1,0} parameter(0)
  %copy.3 = f32[1,20,257]{1,2,0} copy(f32[1,20,257]{2,1,0} %param_0)
  %param_1 = f32[31,257,136]{2,1,0} parameter(1)
  %copy.4 = f32[31,257,136]{0,2,1} copy(f32[31,257,136]{2,1,0} %param_1)
  %custom-call.1 = (f32[1,23,136]{1,2,0}, u8[0]{0}) custom-call(f32[1,20,257]{1,2,0} %copy.3, f32[31,257,136]{0,2,1} %copy.4), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config="{conv_result_scale:1}"
  %get-tuple-element.2 = f32[1,23,136]{1,2,0} get-tuple-element((f32[1,23,136]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %copy.5 = f32[1,23,136]{2,1,0} copy(f32[1,23,136]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[1,23,136]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[1,23,136]{2,1,0}, u8[0]{0}) tuple(f32[1,23,136]{2,1,0} %copy.5, u8[0]{0} %get-tuple-element.3)
})");
}

TEST_F(GpuConvolutionRegressionTest, Computation2) {
  CheckForHloText(R"(
HloModule TestModule

%TestComputation3 (param_0: f32[138,20,1], param_1: f32[31,1,1]) -> (f32[138,23,1], u8[0]) {
  %param_0 = f32[138,20,1]{2,1,0} parameter(0)
  %bitcast = f32[138,20,1]{1,2,0} bitcast(f32[138,20,1]{2,1,0} %param_0)
  %param_1 = f32[31,1,1]{2,1,0} parameter(1)
  %bitcast.1 = f32[31,1,1]{0,2,1} bitcast(f32[31,1,1]{2,1,0} %param_1)
  %custom-call.1 = (f32[138,23,1]{1,2,0}, u8[0]{0}) custom-call(f32[138,20,1]{1,2,0} %bitcast, f32[31,1,1]{0,2,1} %bitcast.1), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config="{conv_result_scale:1}"
  %get-tuple-element.2 = f32[138,23,1]{1,2,0} get-tuple-element((f32[138,23,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %bitcast.2 = f32[138,23,1]{2,1,0} bitcast(f32[138,23,1]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[138,23,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[138,23,1]{2,1,0}, u8[0]{0}) tuple(f32[138,23,1]{2,1,0} %bitcast.2, u8[0]{0} %get-tuple-element.3)
})");
}

TEST_F(GpuConvolutionRegressionTest, Computation3) {
  CheckForHloText(R"(
HloModule TestModule

%TestComputation5 (param_0: f32[138,100,136], param_1: f32[31,136,1]) -> (f32[138,183,1], u8[0]) {
  %param_0 = f32[138,100,136]{2,1,0} parameter(0)
  %copy.3 = f32[138,100,136]{1,2,0} copy(f32[138,100,136]{2,1,0} %param_0)
  %param_1 = f32[31,136,1]{2,1,0} parameter(1)
  %copy.4 = f32[31,136,1]{0,2,1} copy(f32[31,136,1]{2,1,0} %param_1)
  %custom-call.1 = (f32[138,183,1]{1,2,0}, u8[0]{0}) custom-call(f32[138,100,136]{1,2,0} %copy.3, f32[31,136,1]{0,2,1} %copy.4), window={size=31 stride=2 pad=23_23}, dim_labels=b0f_0oi->b0f, custom_call_target="__cudnn$convBackwardInput", backend_config="{conv_result_scale:1}"
  %get-tuple-element.2 = f32[138,183,1]{1,2,0} get-tuple-element((f32[138,183,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=0
  %bitcast = f32[138,183,1]{2,1,0} bitcast(f32[138,183,1]{1,2,0} %get-tuple-element.2)
  %get-tuple-element.3 = u8[0]{0} get-tuple-element((f32[138,183,1]{1,2,0}, u8[0]{0}) %custom-call.1), index=1
  ROOT %tuple.1 = (f32[138,183,1]{2,1,0}, u8[0]{0}) tuple(f32[138,183,1]{2,1,0} %bitcast, u8[0]{0} %get-tuple-element.3)
})");
}

TEST_F(GpuConvolutionRegressionTest, BackwardFilterAlgo0Incorrect) {
  CheckForHloText(R"(
HloModule TestModule

ENTRY %TestComputation {
  %param_0 = f16[7680,96,6,6]{1,3,2,0} parameter(0)
  %param_1 = f16[7680,64,4,4]{1,3,2,0} parameter(1)
  ROOT %custom-call.1 = (f16[64,96,3,3]{1,3,2,0}, u8[0]{0}) custom-call(f16[7680,96,6,6]{1,3,2,0} %param_0, f16[7680,64,4,4]{1,3,2,0} %param_1), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBackwardFilter", backend_config="{conv_result_scale:1}"
})");
}

// See b/135429938.
TEST_F(GpuConvolutionRegressionTest, RedzoneCheckerFailure1) {
  CheckForHloText(R"(
HloModule TestModule

ENTRY %TestComputation {
  %param_0 = f32[2,128,1,378]{3,2,1,0} parameter(0)
  %param_1 = f32[1,5,128,128]{1,0,2,3} parameter(1)
  ROOT %custom-call.1 = (f32[2,128,1,378]{3,2,1,0}, u8[0]{0}) custom-call(%param_0, %param_1), window={size=1x5 pad=0_0x2_2}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})");
}

TEST_F(GpuConvolutionRegressionTest, Conv0D) {
  CheckForHloText(R"(
HloModule TestModule

ENTRY TestComputation {
  %parameter.1 = f32[10,5]{1,0} parameter(0)
  %parameter.2 = f32[5,7]{0,1} parameter(1)
  ROOT %custom-call.1 = (f32[10,7]{1,0}, u8[0]{0}) custom-call(f32[10,5]{1,0} %parameter.1, f32[5,7]{0,1} %parameter.2), window={}, dim_labels=bf_io->bf, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
