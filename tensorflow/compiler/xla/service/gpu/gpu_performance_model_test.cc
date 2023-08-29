/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_performance_model.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class GpuPerformanceModelTest : public HloTestBase {
  GpuHloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

  GpuDeviceInfo dev_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};

 public:
  GpuHloCostAnalysis::Options options_{ShapeSizeBytesFunction(),
                                       /*per_second_rates=*/{},
                                       /*count_multiple_input_accesses=*/true};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  GpuHloCostAnalysis analysis_{options_, &dev_info_};
  GpuPerformanceModelTest() : HloTestBase() {}
};

TEST_F(GpuPerformanceModelTest, LargeWrite) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  c0 = f32[] constant(0)
  ROOT b0 = f32[10000000] broadcast(c0)
}

ENTRY e {
  ROOT r.1 = f32[10000000] fusion(), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_);
  // Dominated by the DRAM bandwidth.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 57, 10);
}

TEST_F(GpuPerformanceModelTest, SmallReadWrite) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = f32[1000] parameter(0)
  p1 = f32[1000] parameter(1)
  ROOT b0 = f32[1000] add(p0, p1)
}

ENTRY e {
  p0 = f32[1000] parameter(0)
  p1 = f32[1000] parameter(1)
  ROOT r.1 = f32[1000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_);
  // Dominated by the kernel launch overhead.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 2, 1);

  GpuPerformanceModel::RecordEstimatedRunTime(root, &analysis_);
  double recorded_cycles = root->backend_config<FusionBackendConfig>()
                               ->reification_cost()
                               .end_to_end_cycles();
  EXPECT_NEAR(recorded_cycles, 8.1, 0.1);
}

TEST_F(GpuPerformanceModelTest, LargeReadWrite) {
  absl::string_view hlo_string = R"(
HloModule m

f {
 p0 = f32[10000000] parameter(0)
 p1 = f32[10000000] parameter(1)
 ROOT a0 = f32[10000000] add(p0, p1)
}

ENTRY e {
 p0 = f32[10000000] parameter(0)
 p1 = f32[10000000] parameter(1)
 ROOT r.1 = f32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_);
  // Dominated by the DRAM bandwidth.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 175, 30);

  GpuPerformanceModel::RecordEstimatedRunTime(root, &analysis_);
  double recorded_cycles = root->backend_config<FusionBackendConfig>()
                               ->reification_cost()
                               .end_to_end_cycles();
  EXPECT_NEAR(recorded_cycles, 220284, 100);
}

TEST_F(GpuPerformanceModelTest, L1CacheEffect) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = f32[10000] parameter(0)
  bc0 = f32[10000,1000] broadcast(p0), dimensions={0}
  b0 = f32[10000000] bitcast(bc0)
  p1 = f32[10000000] parameter(1)
  ROOT a0 = f32[10000000] add(b0, p1)
}

ENTRY e {
  p0 = f32[10000] parameter(0)
  p1 = f32[10000000] parameter(1)
  ROOT r.1 = f32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_);
  // Parameter 0 read is accelerated by L1 cache even though the total data
  // volume is the same as in the test LargeReadWrite above.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 118, 12);
}

TEST_F(GpuPerformanceModelTest, L2CacheEffect) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = f32[1000000] parameter(0)
  bc0 = f32[1000000,10] broadcast(p0), dimensions={0}
  b0 = f32[10000000] bitcast(bc0)
  p1 = f32[10000000] parameter(1)
  ROOT a0 = f32[10000000] add(b0, p1)
}

ENTRY e {
  p0 = f32[1000000] parameter(0)
  p1 = f32[10000000] parameter(1)
  ROOT r.1 = f32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_);
  // Parameter 0 read is accelerated by L2 cache (does not fit in L1).
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 123, 12);
}

TEST_F(GpuPerformanceModelTest, UnusedParameter) {
  Shape shape = ShapeUtil::MakeShape(F32, {100000});

  auto module = std::make_unique<HloModule>("m", HloModuleConfig{});
  HloComputation::Builder b("b");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));

  HloComputation::Builder sub_builder("subcomp");
  HloInstruction* p0f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0f"));
  // p1f is not used.
  HloInstruction* p1f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1f"));
  ASSERT_NE(p1f, nullptr);
  sub_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0f));

  HloComputation* subcomp = module->AddEmbeddedComputation(sub_builder.Build());
  auto fusion = HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {p0, p1}, subcomp);
  b.AddInstruction(std::move(fusion));
  module->AddEntryComputation(b.Build());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 2, 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
