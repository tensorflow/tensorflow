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

#include "xla/service/gpu/model/gpu_performance_model.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"

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

 public:
  GpuHloCostAnalysis::Options options_{ShapeSizeBytesFunction(),
                                       /*per_second_rates=*/{},
                                       /*count_multiple_input_accesses=*/true};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  se::DeviceDescription dev_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
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
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 5, 1);

  GpuPerformanceModel::RecordEstimatedRunTime(root, &analysis_);
  double recorded_cycles = root->backend_config<FusionBackendConfig>()
                               ->reification_cost()
                               .end_to_end_cycles();
  EXPECT_NEAR(recorded_cycles, 257.7, 0.1);
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
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 5, 1);
}

using GpuPerformanceWithCollectiveModelTest = GpuPerformanceModelTest;

TEST_F(GpuPerformanceWithCollectiveModelTest, TestNvmlLibraryLoading) {
#if GOOGLE_CUDA
  EXPECT_TRUE(GpuPerformanceWithCollectiveModel::InitNvml());
  // After successful init, we try to use one of the
  // nvml functions to see if the result is good.
  nvmlDevice_t nvml_device;
  nvmlReturn_t get_device_result =
      xla_nvmlDeviceGetHandleByIndex(0, &nvml_device);
  EXPECT_TRUE(get_device_result == NVML_SUCCESS);

  EXPECT_TRUE(GpuPerformanceWithCollectiveModel::InitNvml());

#endif  // GOOGLE_CUDA
}

TEST_F(GpuPerformanceModelTest, ComputeBoundReducesWithSameLaunchDimensions) {
  // We compare two compute-bound reduces that do ~the same amount of compute
  // and have the same launch dimensions. The result should be approximately
  // the same runtime.
  // TODO(csigg): Once we take occupancy into account for memory bandwidth, we
  // can make this more realistic.
  absl::string_view small_large_reduce_hlo = R"(
HloModule testmodule

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  log0 = f32[] log(p0)
  log1 = f32[] log(log0)
  log2 = f32[] log(log1)
  log3 = f32[] log(log2)
  log4 = f32[] log(log3)
  ROOT max = f32[] maximum(log4, p1)
}

ENTRY fusion {
  c = f32[] constant(-inf)
  p0 = f32[150,32,128] parameter(0)
  reduce.1 = f32[150,32] reduce(p0, c), dimensions={2}, to_apply=max
  ROOT reduce.2 = f32[150] reduce(reduce.1, c), dimensions={1}, to_apply=max
}
)";

  absl::string_view large_small_reduce_hlo = R"(
HloModule testmodule

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  log0 = f32[] log(p0)
  log1 = f32[] log(log0)
  log2 = f32[] log(log1)
  log3 = f32[] log(log2)
  log4 = f32[] log(log3)
  ROOT max = f32[] maximum(log4, p1)
}

ENTRY fusion {
  c = f32[] constant(-inf)
  p0 = f32[150,128,32] parameter(0)
  reduce.1 = f32[150,128] reduce(p0, c), dimensions={2}, to_apply=max
  ROOT reduce.2 = f32[150] reduce(reduce.1, c), dimensions={1}, to_apply=max
}
)";

  auto run = [&](absl::string_view hlo_text)
      -> StatusOr<GpuPerformanceModel::RunTimes> {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_text));
    GpuHloCostAnalysis analysis(options_, &dev_info_);
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&analysis));

    auto* producer =
        module->entry_computation()->GetInstructionWithName("reduce.1");
    std::vector<HloInstruction*> consumers{
        module->entry_computation()->GetInstructionWithName("reduce.2")};

    return GpuPerformanceModel::EstimateRunTimes(producer, &analysis,
                                                 consumers);
  };

  TF_ASSERT_OK_AND_ASSIGN(auto large_small_reduce_runtime,
                          run(small_large_reduce_hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto small_large_reduce_runtime,
                          run(large_small_reduce_hlo));

  // Ignoring memory access patterns and occupancy, the runtime should be about
  // the same.
  EXPECT_NEAR(absl::ToInt64Microseconds(large_small_reduce_runtime.time_fused),
              absl::ToInt64Microseconds(small_large_reduce_runtime.time_fused),
              2);
}

TEST_F(GpuPerformanceModelTest, FusingTransposeIntoReduceIsSlow) {
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

ENTRY fusion {
  c = f32[] constant(-inf)
  p0 = f32[1500,32,128] parameter(0)
  transpose.1 = f32[1500,128,32] transpose(p0), dimensions={0,2,1}
  ROOT reduce.1 = f32[1500,32] reduce(transpose.1, c), dimensions={1}, to_apply=max
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer =
      module->entry_computation()->GetInstructionWithName("transpose.1");
  std::vector<HloInstruction*> consumers{
      module->entry_computation()->GetInstructionWithName("reduce.1")};
  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(producer, &analysis_, consumers);

  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 105, 10);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_fused), 1030, 10);
}

TEST_F(GpuPerformanceModelTest, DusScalesWithUpdates) {
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

fusion.1 {
  p0 = f32[1073741824] parameter(0)
  p1 = f32[1024,1048576] parameter(1)
  p2 = s32[] parameter(2)
  c0 = f32[] constant(0)

  r = f32[1024] reduce(p1, c0), dimensions={1}, to_apply=max
  ROOT dus.1 = f32[1073741824] dynamic-update-slice(p0, r, p2)
}

fusion.2 {
  p0 = f32[1024] parameter(0)
  p1 = f32[1024,1048576] parameter(1)
  p2 = s32[] parameter(2)
  c0 = f32[] constant(0)

  r = f32[1024] reduce(p1, c0), dimensions={1}, to_apply=max
  ROOT dus.1 = f32[1024] dynamic-update-slice(p0, r, p2)
}

ENTRY main {
  p0 = f32[1073741824] parameter(0)
  p1 = f32[1024,1048576] parameter(1)
  p2 = s32[] parameter(2)
  p3 = f32[1024] parameter(3)

  dus1 = f32[1073741824] fusion(p0, p1, p2), kind=kInput, calls=fusion.1
  dus2 = f32[1024] fusion(p3, p1, p2), kind=kInput, calls=fusion.2

  ROOT tuple = (f32[1073741824], f32[1024]) tuple(dus1, dus2)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t1 = GpuPerformanceModel::EstimateRunTimes(
      module->entry_computation()->root_instruction()->operand(0), &analysis_);
  GpuPerformanceModel::RunTimes t2 = GpuPerformanceModel::EstimateRunTimes(
      module->entry_computation()->root_instruction()->operand(1), &analysis_);

  // DUS scales with the size of the updates, so these two fusions should have
  // the same cost.
  EXPECT_NEAR(absl::ToInt64Microseconds(t1.time_unfused),
              absl::ToInt64Microseconds(t2.time_unfused), 10);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
