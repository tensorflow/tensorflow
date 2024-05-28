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

#include "xla/service/gpu/model/gpu_performance_model_base.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuPerformanceModelBaseTest : public HloTestBase {
 public:
  GpuHloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

  GpuHloCostAnalysis::Options options_{ShapeSizeBytesFunction(),
                                       /*per_second_rates=*/{},
                                       /*count_multiple_input_accesses=*/true};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  GpuHloCostAnalysis analysis_{options_, &device_info_};

  GpuPerformanceModelBaseTest() : HloTestBase() {}
};

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_InPlaceDUS) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  param_1 = f32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  log = f32[4,4] log(param_1)
  ROOT dynamic-update-slice = f32[8,16] dynamic-update-slice(param_0, log, c_0, c_0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(&analysis_));

  auto dus_consumer = computation->root_instruction();
  auto log_producer = dus_consumer->mutable_operand(1);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        &analysis_, log_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(0)), 0);
  EXPECT_EQ(get_shared_operand_bytes_accessed(log_producer->operand(0)), 64);
}

TEST_F(GpuPerformanceModelBaseTest, SharedOperandBytesAccessed_DUS) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY entry_computation {
  param_0 = f32[8,16] parameter(0)
  param_1 = f32[4,4] parameter(1)
  c_0 = s32[] constant(0)
  log = f32[8,16] log(param_0)
  ROOT dynamic-update-slice = f32[8,16] dynamic-update-slice(log, param_1, c_0, c_0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(&analysis_));

  auto dus_consumer = computation->root_instruction();
  auto log_producer = dus_consumer->mutable_operand(0);

  auto get_shared_operand_bytes_accessed = [&](const HloInstruction* operand) {
    return GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
        &analysis_, log_producer, dus_consumer, operand);
  };

  EXPECT_EQ(get_shared_operand_bytes_accessed(dus_consumer->operand(1)), 64);
  EXPECT_EQ(get_shared_operand_bytes_accessed(log_producer->operand(0)), 448);
}

// This test documents current behaviour. See comments below how the correct
// result should look like.
TEST_F(GpuPerformanceModelBaseTest,
       ReduceBroadcastedDim_IncorrectBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule m

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

f1 {
  p0 = f32[128] parameter(0)
  c0 = f32[] constant(0)
  broadcast = f32[128,256] broadcast(p0), dimensions={0}
  ROOT reduce = f32[128] reduce(broadcast, c0), dimensions={1}, to_apply=add
}

ENTRY entry_computation {
  param_0 = f32[128] parameter(0)
  param_1 = f32[4,4] parameter(1)
  ROOT fusion = f32[128] fusion(param_0), kind=kLoop, calls=f1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(&analysis_));

  auto root = computation->root_instruction();

  // Cost Model estimates that input element we be re-read in reduce. Each
  // element of reduce output needs only one input element. Bytes accessed
  // should be 4*128=512.
  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(&analysis_, root,
                                                             root->operand(0)),
            /*4*128*256=*/131072);
}

// This test documents current behaviour. See comments below how the correct
// result should look like.
TEST_F(GpuPerformanceModelBaseTest, ElementwiseBitcast_IncorrectBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = f32[128] parameter(0)
  bitcast.1 = f32[8,16] bitcast(p0)
  log = f32[128] log(p0)
  bitcast.2 = f32[8,16] bitcast(log)
  ROOT add = f32[8,16] add(bitcast.1, bitcast.2)
}

ENTRY entry_computation {
  param_0 = f32[128] parameter(0)
  ROOT fusion = f32[8,16] fusion(param_0), kind=kLoop, calls=f1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto computation = module->entry_computation();
  ASSERT_IS_OK(computation->Accept(&analysis_));

  auto root = computation->root_instruction();

  // Bitcast breaks the chain of elementwise utilization even if the bitcast
  // doesn't change physical layout. Each element of `param_0` should be read
  // only once, but Cost Model estimates that it will be accessed twice. Bytes
  // accessed should be 4*128=512.
  EXPECT_EQ(GpuPerformanceModelBase::GetOperandBytesAccessed(&analysis_, root,
                                                             root->operand(0)),
            /*2*4*128=*/1024);
}

TEST_F(GpuPerformanceModelBaseTest, EstimateFusionLaunchDimensions_LoopFusion) {
  absl::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = f32[8,16,128] parameter(0)
  log = f32[8,16,128] log(p0)
  ROOT add = f32[8,16,128] add(p0, log)
}

ENTRY entry_computation {
  param_0 = f32[8,16,128] parameter(0)
  ROOT fusion = f32[8,16,128] fusion(param_0), kind=kLoop, calls=f1
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = AnalyzeFusion(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  EXPECT_EQ(launch_dimensions.num_blocks(), 16);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 1024);
}

TEST_F(GpuPerformanceModelBaseTest,
       EstimateFusionLaunchDimensions_TritonSoftMaxFusion) {
  absl::string_view hlo_string = R"(
max {
  p1 = f32[] parameter(1)
  p0 = f32[] parameter(0)
  ROOT m = f32[] maximum(p0, p1)
}

triton_softmax_computation {
  p0 = f32[16,970] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[16] reduce(p0, constant), dimensions={1}, to_apply=max
  broadcast = f32[16,970] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[16,970] subtract(p0, broadcast)
}

ENTRY e {
  p0 = f32[16,970]{1,0} parameter(0)
  ROOT r = f32[16,970]{1,0} fusion(p0), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={"fusion_backend_config": {kind: "__triton_softmax"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = AnalyzeFusion(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  EXPECT_EQ(launch_dimensions.num_blocks(), 16);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 64);
}

TEST_F(GpuPerformanceModelBaseTest,
       EstimateFusionLaunchDimensions_CudnnFusion) {
  absl::string_view hlo_string = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,256] parameter(1)
  ROOT r = f32[32,256] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,256] parameter(1)
  ROOT _ = f32[32,256] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto fusion_analysis = AnalyzeFusion(
      *module->entry_computation()->root_instruction(), device_info_);
  auto launch_dimensions =
      GpuPerformanceModelBase::EstimateFusionLaunchDimensions(fusion_analysis);

  // CuNnnFusion doesn't implement KernelLaunchInsterface, so
  // EstimateFusionLaunchDimensions returns a default estimate.
  EXPECT_EQ(launch_dimensions.num_blocks(), 64);
  EXPECT_EQ(launch_dimensions.num_threads_per_block(), 128);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
