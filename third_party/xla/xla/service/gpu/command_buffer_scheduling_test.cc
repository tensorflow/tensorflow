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
#include "xla/service/gpu/command_buffer_scheduling.h"

#include <gtest/gtest.h>
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {

namespace {

class CommandBufferSchedulingTest : public HloTestBase {};

TEST_F(CommandBufferSchedulingTest, SingleFusion) {
  const char* hlo = R"(
      HloModule TestModule

      %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        ROOT %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
      })";

  RunAndFilecheckHloRewrite(hlo, CommandBufferScheduling(), R"(
// CHECK: %command_buffer (param: s32[], param.1: s32[]) -> s32[] {
// CHECK:   %param = s32[] parameter(0)
// CHECK:   %param.1 = s32[] parameter(1)
// CHECK:   ROOT %fusion.1 = s32[] fusion(%param, %param.1), kind=kLoop, calls=%fused_computation
// CHECK: }
//
// CHECK: ENTRY %main (a: s32[], b: s32[]) -> s32[] {
// CHECK:   %a = s32[] parameter(0)
// CHECK:   %b = s32[] parameter(1)
// CHECK:   ROOT %call = s32[] call(%a, %b), to_apply=%command_buffer
// CHECK: })");
}

}  // namespace

}  // namespace xla::gpu
