/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/transforms/fusion_wrapper.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

auto MakeDeviceDescription() {
  stream_executor::DeviceDescription device_description{
      stream_executor::GpuDeviceInfoProto{}};
  device_description.set_threads_per_warp(32);
  return device_description;
}

class FusionWrapperTest : public HloTestBase {
 public:
  using HloTestBase::HloTestBase;

  const stream_executor::DeviceDescription& device_description() const {
    return device_description_;
  }

 private:
  const stream_executor::DeviceDescription device_description_{
      MakeDeviceDescription()};
};

TEST_F(FusionWrapperTest, ConvolutionWorks) {
  RunAndFilecheckHloRewrite(R"(HloModule TestModule

ENTRY TestComputation {
  input = f32[1,10,1,10,5,20]{5,4,3,2,1,0} parameter(0)
  kernel = f32[20,1,2,1,4,15]{5,4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,9,1,7,5]{5,4,3,2,1,0} convolution(input, kernel), dim_labels=0123bf_i0123o->f0123b, window={size=1x2x1x4}
})",
                            FusionWrapper(device_description()), R"(
// CHECK: %wrapped_convolution_computation (param_0: f32[1,10,1,10,5,20], param_1: f32[20,1,2,1,4,15]) -> f32[15,1,9,1,7,5] {
// CHECK:   %param_0 = f32[1,10,1,10,5,20]{5,4,3,2,1,0} parameter(0)
// CHECK:   %param_1 = f32[20,1,2,1,4,15]{5,4,3,2,1,0} parameter(1)
// CHECK:   ROOT %conv.1 = f32[15,1,9,1,7,5]{5,4,3,2,1,0} convolution(%param_0, %param_1), window={size=1x2x1x4}, dim_labels=0123bf_i0123o->f0123b
// CHECK: }

// CHECK: ENTRY %TestComputation (input: f32[1,10,1,10,5,20], kernel: f32[20,1,2,1,4,15]) ->  f32[15,1,9,1,7,5] {
// CHECK:   %input = f32[1,10,1,10,5,20]{5,4,3,2,1,0} parameter(0)
// CHECK:   %kernel = f32[20,1,2,1,4,15]{5,4,3,2,1,0} parameter(1)
// CHECK:   ROOT %wrapped_convolution = f32[15,1,9,1,7,5]{5,4,3,2,1,0} fusion(%input, %kernel), kind=kLoop, calls=%wrapped_convolution_computation
// CHECK: })");
}

TEST_F(FusionWrapperTest, SimpleOp) {
  RunAndFilecheckHloRewrite(R"(
      HloModule TestModule

      ENTRY TestComputation {
        p0 = f16[30,41] parameter(0)
        p1 = f16[30,41] parameter(1)
        ROOT result = f16[60, 41] concatenate(p0, p1), dimensions={0}
      })",
                            FusionWrapper(device_description()), R"(
// CHECK: %wrapped_concatenate_computation (param_0: f16[30,41], param_1: f16[30,41]) -> f16[60,41] {
// CHECK:   %param_0 = f16[30,41]{1,0} parameter(0)
// CHECK:   %param_1 = f16[30,41]{1,0} parameter(1)
// CHECK:   ROOT %result.1 = f16[60,41]{1,0} concatenate(%param_0, %param_1), dimensions={0}
// CHECK: }

// CHECK: ENTRY %TestComputation (p0: f16[30,41], p1: f16[30,41]) -> f16[60,41] {
// CHECK:   %p0 = f16[30,41]{1,0} parameter(0)
// CHECK:   %p1 = f16[30,41]{1,0} parameter(1)
// CHECK:   ROOT %wrapped_concatenate = f16[60,41]{1,0} fusion(%p0, %p1), kind=kLoop, calls=%wrapped_concatenate_computation
// CHECK: })");
}

TEST_F(FusionWrapperTest, Scatter) {
  RunAndFilecheckHloRewrite(R"(
      HloModule ScatterIntoScalar

      update_s32 {
        lhs = s32[] parameter(0)
        ROOT rhs = s32[] parameter(1)
      }

      ENTRY main {
        parameter.1 = s32[] parameter(0)
        parameter.2 = s32[0]{0} parameter(1)
        parameter.3 = s32[] parameter(2)
        ROOT scatter_ScatterIntoScalar = s32[] scatter(parameter.1, parameter.2, parameter.3),
            update_window_dims={},
            inserted_window_dims={},
            scatter_dims_to_operand_dims={},
            index_vector_dim=0,
            to_apply=update_s32
      })",
                            FusionWrapper(device_description()), R"(
// CHECK: wrapped_scatter_computation
// CHECK:   %[[param_0:.*]] = s32[] parameter(0)
// CHECK:   %[[param_1:.*]] = s32[0]{0} parameter(1)
// CHECK:   %[[param_2:.*]] = s32[] parameter(2)
// CHECK:   ROOT %{{.*}} = s32[] scatter(%[[param_0]], %[[param_1]], %[[param_2]])

// CHECK: ENTRY
// CHECK:   %[[p0:.*]] = s32[] parameter(0)
// CHECK:   %[[p1:.*]] = s32[0]{0} parameter(1)
// CHECK:   %[[p2:.*]] = s32[] parameter(2)
// CHECK:   ROOT %{{.*}} = s32[] fusion(%[[p0]], %[[p1]], %[[p2]]), kind=kInput, calls=%wrapped_scatter_computation
// CHECK: })");
}

TEST_F(FusionWrapperTest, ControlDependency) {
  RunAndFilecheckHloRewrite(R"(
      HloModule TestModule

      fusion {
        ROOT param = f32[] parameter(0)
      }

      ENTRY main {
        param = f32[] parameter(0)
        fusion = f32[] fusion(param), kind=kLoop, calls=fusion
        constant_one = f32[] constant(1)
        ROOT add = f32[] add(param, constant_one), control-predecessors={fusion}
      })",
                            FusionWrapper(device_description()), R"(
// CHECK: ROOT %wrapped_add = f32[] fusion(%param.1, %constant_one),
// CHECK-SAME: control-predecessors={%fusion})");
}

TEST_F(FusionWrapperTest, Copy) {
  // Avoid rewriting copies, so that the rematerialization pass
  // can avoid rematerializing copies inserted by copy-insertion
  // (the rematerialization could read overwritten data).
  RunAndFilecheckHloRewrite(R"(
      HloModule Copy

      ENTRY %main (parameter.1: f32[5]) -> f32[5] {
        %parameter.1 = f32[5]{0} parameter(0)
        ROOT %copy.3 = f32[5]{0} copy(f32[5]{0} %parameter.1)
      })",
                            FusionWrapper(device_description()),
                            // No change
                            std::nullopt);
}

TEST_F(FusionWrapperTest, While) {
  RunAndFilecheckHloRewrite(R"(
      HloModule While

      %body {
        %parameter.5 = (f32[5]{0}) parameter(0)
        %constant_8 = f32[] constant(0)
        %broadcast.9 = f32[5]{0} broadcast(f32[] %constant_8), dimensions={}
        ROOT %tuple.2 = (f32[5]{0}) tuple(f32[5]{0} %broadcast.9)
      }

      %cond {
        %parameter.12 = (f32[5]{0}) parameter(0)
        ROOT %constant_1 = pred[] constant(false)
      }

      ENTRY %main (parameter.1: f32[5]) -> (f32[5]) {
        %parameter.1 = f32[5]{0} parameter(0)
        %copy.3 = f32[5]{0} copy(f32[5]{0} %parameter.1)
        %tuple = (f32[5]{0}) tuple(f32[5]{0} %copy.3)
        ROOT %while.19 = (f32[5]{0}) while((f32[5]{0}) %tuple), condition=%cond, body=%body
      })",
                            FusionWrapper(device_description()), R"(
// CHECK: %wrapped_broadcast_computation {{.*}} {
// CHECK:  %param_0 = f32[] parameter(0)
// CHECK:  ROOT %broadcast.0 = f32[5]{0} broadcast(%param_0), dimensions={}
// CHECK: }
// CHECK: %body {{.*}} {
// CHECK:   %parameter.5 = (f32[5]{0}) parameter(0)
// CHECK:   %constant_8 = f32[] constant(0)
// CHECK:   %wrapped_broadcast = f32[5]{0} fusion(%constant_8), kind=kLoop, calls=%wrapped_broadcast_computation
// CHECK:   ROOT %tuple.2 = (f32[5]{0}) tuple(%wrapped_broadcast)
// CHECK: }
// CHECK: %cond {{.*}} {
// CHECK:   %parameter.12 = (f32[5]{0}) parameter(0)
// CHECK:   ROOT %constant_1 = pred[] constant(false)
// CHECK: }
// CHECK: ENTRY %main {{.*}} {
// CHECK:   %parameter.1 = f32[5]{0} parameter(0)
// CHECK:   %copy.3 = f32[5]{0} copy(%parameter.1)
// CHECK:   %tuple = (f32[5]{0}) tuple(%copy.3)
// CHECK:   ROOT %while.19 = (f32[5]{0}) while(%tuple), condition=%cond, body=%body
// CHECK: })");
}

TEST_F(FusionWrapperTest, WhileInFusion) {
  RunAndFilecheckHloRewrite(R"(
      HloModule While

      %body {
        %parameter.5 = (f32[5]{0}) parameter(0)
        %constant_8 = f32[] constant(0)
        %broadcast.9 = f32[5]{0} broadcast(f32[] %constant_8), dimensions={}
        ROOT %tuple.2 = (f32[5]{0}) tuple(f32[5]{0} %broadcast.9)
      }

      %cond {
        %parameter.12 = (f32[5]{0}) parameter(0)
        ROOT %constant_1 = pred[] constant(false)
      }

      %fusion {
        %parameter.1 = f32[5]{0} parameter(0)
        %copy.3 = f32[5]{0} copy(f32[5]{0} %parameter.1)
        %tuple = (f32[5]{0}) tuple(f32[5]{0} %copy.3)
        ROOT %while.19 = (f32[5]{0}) while((f32[5]{0}) %tuple), condition=%cond, body=%body
      }

      ENTRY %main (parameter.1: f32[5]) -> (f32[5]) {
        %parameter.1 = f32[5]{0} parameter(0)
        ROOT %fusion = (f32[5]{0}) fusion(f32[5]{0} %parameter.1), kind=kLoop, calls=%fusion
      })",
                            FusionWrapper(device_description()),
                            // No change
                            std::nullopt);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
