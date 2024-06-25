/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/triton.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

class TritonFusionTest : public HloTestBase {};

TEST_F(TritonFusionTest, TritonSoftmaxFusion) {
#ifndef GOOGLE_CUDA
  GTEST_SKIP() << "Triton fusion only enable for CUDA devices.";
#endif

  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule t

    add {
      Arg_0 = f32[] parameter(0)
      Arg_1 = f32[] parameter(1)
      ROOT add = f32[] add(Arg_0, Arg_1)
    }

    auxiliary_computation {
      parameter_0 = f32[125]{0} parameter(0)
      ROOT broadcast = f32[125,127]{1,0} broadcast(parameter_0), dimensions={0}
    }

    triton_softmax_computation {
      parameter_0 = f32[125,127]{1,0} parameter(0)
      multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
      constant_0 = f32[] constant(0)
      reduce_0 = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
      broadcast_4 = f32[125,127]{1,0} broadcast(reduce_0), dimensions={0}
      ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast_4)
    }

    ENTRY main {
      param_0 = f32[125]{0} parameter(0)
      auxiliary_fusion = f32[125,127]{1,0} fusion(param_0), kind=kLoop, calls=auxiliary_computation
      ROOT triton_softmax = f32[125,127]{1,0} fusion(auxiliary_fusion), kind=kCustom, calls=triton_softmax_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
      })")
                    .value();

  stream_executor::GpuDeviceInfoProto device_info_proto;
  stream_executor::DeviceDescription device_info(device_info_proto);

  auto* root = module->entry_computation()->root_instruction();
  auto analysis_fused =
      AnalyzeProducerConsumerFusion(*root->operand(0), *root, device_info);

  auto emitter_fused =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_fused});
  auto triton_fusion = dynamic_cast<TritonFusion*>(emitter_fused.get());
  ASSERT_NE(triton_fusion, nullptr);
  auto launch_config = triton_fusion->launch_config();
  ASSERT_NE(launch_config, std::nullopt);
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 125);
  EXPECT_EQ(launch_config->launch_dimensions.num_threads_per_block(), 32);
  EXPECT_THAT(launch_config->output_tile_sizes, ElementsAre(1, 127));

  auto analysis_consumer = AnalyzeFusion(*root, device_info);

  auto emitter_consumer =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis_consumer});
  ASSERT_NE(dynamic_cast<TritonFusion*>(emitter_consumer.get()), nullptr);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
