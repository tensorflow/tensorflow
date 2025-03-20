/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/performance_info_wrapper.h"

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/utils/hlo_cost_analysis_wrapper.h"
#include "tensorflow/core/profiler/utils/hlo_module_map.h"
#include "tensorflow/core/profiler/utils/xprof_gpu_cost_analysis.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::EqualsProto;
using ::testing::UnorderedElementsAre;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

TEST(PerformanceInfoWrapper, Test16BitPricision) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
  x = bf16[2,4]{1,0} parameter(0)
  y = bf16[2,4]{1,0} parameter(1)
  ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
}
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper =
      CreateXprofGpuCostAnalysis();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis_wrapper->GetXlaCostAnalysis()));
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(cost_analysis_wrapper.get(), root);
  EXPECT_EQ(performance_info_wrapper->DeviceFlops(),
            performance_info_wrapper->ModelFlops());
  EXPECT_EQ(performance_info_wrapper->ComputationalPrimitiveBitwidth(), 16);
  EXPECT_GT(performance_info_wrapper->DeviceFlops(), 0);
}

TEST(PerformanceInfoWrapper, Test4BitPricision) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
  x = s4[2,4]{1,0} parameter(0)
  y = s4[2,4]{1,0} parameter(1)
  ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
}
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper =
      CreateXprofGpuCostAnalysis();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis_wrapper->GetXlaCostAnalysis()));
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(cost_analysis_wrapper.get(), root);
  EXPECT_EQ(performance_info_wrapper->DeviceFlops(),
            performance_info_wrapper->ModelFlops() / 4);
  EXPECT_EQ(performance_info_wrapper->ComputationalPrimitiveBitwidth(), 4);
  EXPECT_GT(performance_info_wrapper->DeviceFlops(), 0);
}

TEST(PerformanceInfoWrapper, TestInputBitwidths) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
  x = s16[2,4]{1,0} parameter(0)
  y = s4[2,4]{1,0} parameter(1)
  ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
}
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper =
      CreateXprofGpuCostAnalysis();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis_wrapper->GetXlaCostAnalysis()));
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(cost_analysis_wrapper.get(), root);
  // Expect the input bitwidths to be 16 and 4 based on the graph created above.
  EXPECT_THAT(performance_info_wrapper->InputBitwidths(),
              UnorderedElementsAre(16, 4));
}

TEST(PerformanceInfoWrapper, TestMemoryAccessed) {
  auto performance_info =
      std::make_unique<PerformanceInfoWrapper::PerfInfoType>();
  tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        flops: 1000000
        bytes_accessed: 100
        memory_accessed_breakdown {
          is_read: true
          memory_space: 1
          bytes_accessed: 200
        }
        memory_accessed_breakdown {
          is_read: false
          memory_space: 1
          bytes_accessed: 300
        }
      )pb",
      performance_info.get());
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(std::move(performance_info));
  EXPECT_THAT(performance_info_wrapper->GetMemmoryAccessBreakdown(),
              IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                memory_accessed {
                  operation_type: READ
                  memory_space: 1
                  bytes_accessed: 200
                }
                memory_accessed {
                  operation_type: WRITE
                  memory_space: 1
                  bytes_accessed: 300
                })pb")));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
