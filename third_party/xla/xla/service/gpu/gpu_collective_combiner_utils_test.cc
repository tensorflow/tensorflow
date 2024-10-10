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

#include "xla/service/gpu/gpu_collective_combiner_utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/collective_utils.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using CollectiveCombinerUtilsTest = HloTestBase;

TEST_F(CollectiveCombinerUtilsTest,
       ComputeSuggestedCombinerThresholdReturnsMemoryThresholdForDeviceInfo) {
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  int pointer_size = 4;
  stream_executor::DeviceDescription device_info;
  device_info.set_device_memory_size(20000);

  int64_t suggested_threshold = ComputeSuggestedCombinerThreshold(
      *module, device_info, gpu::ScheduleGpuModuleWithMemoryScheduler,
      HloOpcode::kAllReduce, pointer_size);

  // device size = 20000 bytes
  // slop factor = 0.95
  // peak memory = parameters + output = (2*32*32 + 32*32) * 4 bytes = 12288
  // suggested thresholds = device size * slop factor - peak memory
  EXPECT_EQ(suggested_threshold, 6712);
}

TEST_F(CollectiveCombinerUtilsTest,
       ComputeSuggestedCombinerThresholdReturnsMemoryThresholdForModuleConfig) {
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_device_memory_size(20000);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloText, config));
  int pointer_size = 4;
  stream_executor::DeviceDescription device_info;

  int64_t suggested_threshold = ComputeSuggestedCombinerThreshold(
      *module, device_info, gpu::ScheduleGpuModuleWithMemoryScheduler,
      HloOpcode::kAllReduce, pointer_size);

  // device size = 20000 bytes
  // slop factor = 0.95
  // peak memory = parameters + output = (2*32*32 + 32*32) * 4 bytes = 12288
  // suggested thresholds = device size * slop factor - peak memory
  EXPECT_EQ(suggested_threshold, 6712);
}

TEST_F(
    CollectiveCombinerUtilsTest,
    ComputeSuggestedCombinerThresholdReturnsDefaultValueUponSchedulingFailure) {  // NOLINT
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  int pointer_size = 4;
  stream_executor::DeviceDescription device_info;
  device_info.set_device_memory_size(20000);

  auto sched_fun = [](const HloModule* m, int64_t p_sz,
                      int64_t* p) -> absl::StatusOr<HloSchedule> {
    return absl::UnimplementedError("Fail.");
  };

  int64_t suggested_threshold_all_reduce = ComputeSuggestedCombinerThreshold(
      *module, device_info, sched_fun, HloOpcode::kAllReduce, pointer_size);
  int64_t suggested_threshold_all_gather = ComputeSuggestedCombinerThreshold(
      *module, device_info, sched_fun, HloOpcode::kAllGather, pointer_size);
  int64_t suggested_threshold_reduce_scatter =
      ComputeSuggestedCombinerThreshold(*module, device_info, sched_fun,
                                        HloOpcode::kReduceScatter,
                                        pointer_size);

  EXPECT_EQ(suggested_threshold_all_reduce, kDefaultAllReduceCombineThreshold);
  EXPECT_EQ(suggested_threshold_all_gather, kDefaultAllGatherCombineThreshold);
  EXPECT_EQ(suggested_threshold_reduce_scatter,
            kDefaultReduceScatterCombineThreshold);
}

}  // namespace
}  // namespace xla::gpu
