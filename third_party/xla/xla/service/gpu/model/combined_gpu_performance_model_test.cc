/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/model/combined_gpu_performance_model.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

class CombinedGpuPerformanceModelTest : public HloHardwareIndependentTestBase {
 public:
  CombinedGpuPerformanceModelTest() : analysis_(options_, device_info_) {
    options_.count_multiple_input_accesses = true;
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  mlir::MLIRContext mlir_context_;
  GpuHloCostAnalysis::Options options_;
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  HloFusionAnalysisCache fusion_analysis_cache_{device_info_};
  GpuHloCostAnalysis analysis_;
  CombinedGpuPerformanceModel model_{
      device_info_, fusion_analysis_cache_, mlir_context_,
      [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); }};
};

TEST_F(CombinedGpuPerformanceModelTest,
       ReturnsGpuPerformanceModelResultForNonTritonFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry_computation {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT add = f32[1024] add(p0, p1)
    }
  )")
                    .value();
  HloInstruction* add = module->entry_computation()->root_instruction();
  ASSERT_OK(add->Accept(&analysis_));
  GpuPerformanceModelCache reference_cache;
  GpuPerformanceModel reference_model(device_info_, fusion_analysis_cache_,
                                      reference_cache, &mlir_context_);
  const EstimateRunTimeData expected_result =
      reference_model.EstimateRunTimeForInstruction(add, &analysis_);

  auto result = model_.EstimateRunTimeForInstruction(add, &analysis_);
  ASSERT_OK(result.status());
  EXPECT_EQ(result->flops, expected_result.flops);
  EXPECT_EQ(result->bytes_read, expected_result.bytes_read);
  EXPECT_EQ(result->bytes_written, expected_result.bytes_written);
  EXPECT_EQ(result->read_time, expected_result.read_time);
  EXPECT_EQ(result->write_time, expected_result.write_time);
  EXPECT_EQ(result->compute_time, expected_result.compute_time);
  EXPECT_EQ(result->exec_time, expected_result.exec_time);
}

TEST_F(CombinedGpuPerformanceModelTest,
       ReturnsIndexingModelResultForTritonFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    triton_fusion {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT add = f32[1024] add(p0, p1)
    }
    ENTRY entry_computation {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT fusion = f32[1024] fusion(p0, p1), kind=kCustom, calls=triton_fusion,
        backend_config={
          "fusion_backend_config": {
            kind: "__triton",
            block_level_fusion_config: {
              output_tiles: [{sizes: ["1024"]}],
              num_warps: "1"
            }
          }
        }
    }
  )")
                    .value();
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK(fusion->Accept(&analysis_));
  GpuPerformanceModelWithIndexingAnalysis reference_model(
      &device_info_, &fusion_analysis_cache_,
      [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); },
      &mlir_context_);
  const EstimateRunTimeData expected_result =
      reference_model.EstimateRunTimeForInstruction(fusion);

  auto result = model_.EstimateRunTimeForInstruction(fusion, &analysis_);

  ASSERT_OK(result.status());
  EXPECT_EQ(result->flops, expected_result.flops);
  EXPECT_EQ(result->bytes_read, expected_result.bytes_read);
  EXPECT_EQ(result->bytes_written, expected_result.bytes_written);
  EXPECT_EQ(result->read_time, expected_result.read_time);
  EXPECT_EQ(result->write_time, expected_result.write_time);
  EXPECT_EQ(result->compute_time, expected_result.compute_time);
  EXPECT_EQ(result->exec_time, expected_result.exec_time);
}

// TODO: b/493907020 Remove this after removing from GpuPerformanceModel.
TEST_F(CombinedGpuPerformanceModelTest,
       EstimateRunTimesMatchesGpuPerformanceModel) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry_computation {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      add = f32[1024] add(p0, p1)
      ROOT exp = f32[1024] exponential(add)
    }
  )")
                    .value();
  HloInstruction* add =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloInstruction* exp = module->entry_computation()->root_instruction();
  ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  GpuPerformanceModelCache cache;
  GpuPerformanceModel standalone_model(device_info_, fusion_analysis_cache_,
                                       cache, &mlir_context_);
  standalone_model.EstimateRunTimeForInstruction(add, &analysis_);
  standalone_model.EstimateRunTimeForInstruction(exp, &analysis_);
  auto expected = standalone_model.EstimateRunTimes(add, &analysis_, {exp});

  auto result = model_.EstimateRunTimes(add, &analysis_, {exp});

  ASSERT_OK(result.status());
  EXPECT_EQ(result->time_unfused, expected.time_unfused);
  EXPECT_EQ(result->time_fused, expected.time_fused);
}

// TODO: b/493907020 Remove this after removing from GpuPerformanceModel.
TEST_F(CombinedGpuPerformanceModelTest,
       EstimateRunTimesForMultiOutputMatchesGpuPerformanceModel) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry_computation {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      add = f32[1024] add(p0, p1)
      exp = f32[1024] exponential(add)
      ROOT tuple = (f32[1024], f32[1024]) tuple(add, exp)
    }
  )")
                    .value();
  HloInstruction* add =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloInstruction* exp =
      module->entry_computation()->root_instruction()->mutable_operand(1);
  ASSERT_OK(module->entry_computation()->Accept(&analysis_));
  GpuPerformanceModelCache cache;
  GpuPerformanceModel standalone_model(device_info_, fusion_analysis_cache_,
                                       cache, &mlir_context_);
  auto expected = standalone_model.EstimateRunTimesForMultiOutputFusion(
      add, exp, &analysis_);

  auto result = model_.EstimateRunTimesForMultiOutput(add, exp, &analysis_);

  ASSERT_OK(result.status());
  EXPECT_EQ(result->time_unfused, expected.time_unfused);
  EXPECT_EQ(result->time_fused, expected.time_fused);
}

TEST_F(CombinedGpuPerformanceModelTest, CachesResults) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry_computation {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT add = f32[1024] add(p0, p1)
    }
  )")
                    .value();
  HloInstruction* add = module->entry_computation()->root_instruction();
  ASSERT_OK(add->Accept(&analysis_));
  EXPECT_FALSE(model_.GetCache().Get(*add).has_value());

  auto result = model_.EstimateRunTimeForInstruction(add, &analysis_);

  ASSERT_OK(result.status());
  EXPECT_TRUE(model_.GetCache().Get(*add).has_value());
  EXPECT_EQ(model_.GetCache().Get(*add)->exec_time, result->exec_time);
}

TEST_F(CombinedGpuPerformanceModelTest, InvalidatesCache) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry_computation {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT add = f32[1024] add(p0, p1)
    }
  )")
                    .value();
  HloInstruction* add = module->entry_computation()->root_instruction();
  ASSERT_OK(add->Accept(&analysis_));
  ASSERT_OK(model_.EstimateRunTimeForInstruction(add, &analysis_).status());
  EXPECT_TRUE(model_.GetCache().Get(*add).has_value());

  model_.Invalidate(*add);

  EXPECT_FALSE(model_.GetCache().Get(*add).has_value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
