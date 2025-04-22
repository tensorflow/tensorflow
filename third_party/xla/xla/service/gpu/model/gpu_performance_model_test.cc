/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuPerformanceModelTest : public HloHardwareIndependentTestBase {
 public:
  GpuPerformanceModel::RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      std::vector<HloInstruction*> fused_consumers = {}) {
    auto config = GpuPerformanceModelOptions::Default(
        &fusion_analysis_cache_, &gpu_performance_model_cache_);

    auto runtime_data = GpuPerformanceModel::EstimateRunTimeForInstruction(
        producer, device_info_, &analysis_, config);
    gpu_performance_model_cache_.Set(*producer, runtime_data);
    for (auto consumer : fused_consumers) {
      auto runtime_data = GpuPerformanceModel::EstimateRunTimeForInstruction(
          consumer, device_info_, &analysis_, config);
      gpu_performance_model_cache_.Set(*consumer, runtime_data);
    }
    return GpuPerformanceModel::EstimateRunTimes(
        producer, device_info_, &analysis_, config, fused_consumers);
  }

  mlir::MLIRContext mlir_context_;
  GpuHloCostAnalysis::Options options_{.count_multiple_input_accesses = true};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  HloFusionAnalysisCache fusion_analysis_cache_{device_info_};
  GpuHloCostAnalysis analysis_{options_, device_info_};
  GpuPerformanceModelCache gpu_performance_model_cache_;

  GpuPerformanceModelWithIndexingAnalysis indexing_cost_model_{
      &device_info_, &fusion_analysis_cache_, HloCostAnalysis::DefaultShapeSize,
      &mlir_context_};
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

  auto t = EstimateRunTimes(root);
  // Dominated by the DRAM bandwidth.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 53, 10);

  auto indexing_t = indexing_cost_model_.EstimateRunTimes(root);
  EXPECT_NEAR(absl::ToInt64Microseconds(indexing_t.time_unfused), 53, 10);
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

  auto t = EstimateRunTimes(root);
  // Dominated by the kernel launch overhead.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 1, 1);

  GpuPerformanceModel::RecordEstimatedRunTime(
      root, device_info_, &analysis_, GpuPerformanceModelOptions::Default());
  auto reification_cost = root->backend_config<GpuBackendConfig>()
                              ->fusion_backend_config()
                              .reification_cost();
  EXPECT_NEAR(reification_cost.end_to_end_cycles(), 38.4, 0.1);
  EXPECT_NEAR(reification_cost.exec_time_us(), 0, 1);

  auto indexing_t = indexing_cost_model_.EstimateRunTimes(root);
  EXPECT_NEAR(absl::ToInt64Microseconds(indexing_t.time_unfused), 1, 1);
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

  auto t = EstimateRunTimes(root);
  // Dominated by the DRAM bandwidth.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 175, 30);

  GpuPerformanceModel::RecordEstimatedRunTime(
      root, device_info_, &analysis_, GpuPerformanceModelOptions::Default());
  auto reification_cost = root->backend_config<GpuBackendConfig>()
                              ->fusion_backend_config()
                              .reification_cost();
  EXPECT_NEAR(reification_cost.end_to_end_cycles(), 220284, 100);
  EXPECT_NEAR(reification_cost.exec_time_us(), 156, 10);
  EXPECT_NEAR(reification_cost.compute_time_us(), 1, 1);
  EXPECT_NEAR(reification_cost.memory_access_time_us(), 156, 10);
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

  auto t = EstimateRunTimes(root);
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

  auto t = EstimateRunTimes(root);
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

  auto t = EstimateRunTimes(root);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 1, 1);
}

TEST_F(GpuPerformanceModelTest, ComputeBoundReducesWithSameLaunchDimensions) {
  // We compare two compute-bound reduces that do ~the same amount of compute
  // and have the same launch dimensions. The result should be approximately
  // the same runtime.
  // TODO(csigg): Once we take occupancy into account for memory bandwidth, we
  // can make this more realistic.
  absl::string_view kHlo = R"(
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
  reduce.2 = f32[150] reduce(reduce.1, c), dimensions={1}, to_apply=max

  p1 = f32[150,128,32] parameter(1)
  reduce.3 = f32[150,128] reduce(p1, c), dimensions={2}, to_apply=max
  reduce.4 = f32[150] reduce(reduce.3, c), dimensions={1}, to_apply=max

  ROOT res = (f32[150], f32[150]) tuple(reduce.2, reduce.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto run = [&](absl::string_view reduce_name_1,
                 absl::string_view reduce_name_2)
      -> absl::StatusOr<GpuPerformanceModel::RunTimes> {
    auto* producer =
        module->entry_computation()->GetInstructionWithName(reduce_name_1);
    std::vector<HloInstruction*> consumers{
        module->entry_computation()->GetInstructionWithName(reduce_name_2)};

    return EstimateRunTimes(producer, consumers);
  };

  TF_ASSERT_OK_AND_ASSIGN(auto large_small_reduce_runtime,
                          run("reduce.1", "reduce.2"));
  TF_ASSERT_OK_AND_ASSIGN(auto small_large_reduce_runtime,
                          run("reduce.3", "reduce.4"));

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

  auto t = EstimateRunTimes(producer, consumers);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 105, 10);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_fused), 514, 10);
}

// Same as FusingTransposeIntoReduceIsSlow, but artificially wrapping the
// transpose in a multi-output fusion with 1 output, to check that we still get
// the same results.
TEST_F(GpuPerformanceModelTest,
       FusingTransposeMultiOutputFusionIntoReduceIsSlow) {
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

transpose_fusion {
  param0 = f32[1500,32,128] parameter(0)
  transpose.1 = f32[1500,128,32] transpose(param0), dimensions={0,2,1}
  ROOT res = (f32[1500,128,32]) tuple(transpose.1)
}

ENTRY fusion {
  c = f32[] constant(-inf)
  p0 = f32[1500,32,128] parameter(0)
  fusion = (f32[1500,128,32]) fusion(p0), kind=kInput, calls=transpose_fusion
  gte = f32[1500,128,32] get-tuple-element(fusion), index=0
  ROOT reduce.1 = f32[1500,32] reduce(gte, c), dimensions={1}, to_apply=max
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer =
      module->entry_computation()->GetInstructionWithName("fusion");
  std::vector<HloInstruction*> consumers{
      module->entry_computation()->GetInstructionWithName("reduce.1")};

  auto t = EstimateRunTimes(producer, consumers);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 105, 10);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_fused), 514, 10);
}

TEST_F(GpuPerformanceModelTest, FusingNonMinorTransposeIntoReduceIsFast) {
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

ENTRY fusion {
  c = f32[] constant(-inf)
  p0 = f32[1500,32,128]{1,2,0} parameter(0)
  transpose.1 = f32[1500,128,32]{2,0,1} transpose(p0), dimensions={0,2,1}
  ROOT reduce.1 = f32[1500,32] reduce(transpose.1, c), dimensions={1}, to_apply=max
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer =
      module->entry_computation()->GetInstructionWithName("transpose.1");
  std::vector<HloInstruction*> consumers{
      module->entry_computation()->GetInstructionWithName("reduce.1")};

  auto t = EstimateRunTimes(producer, consumers);
  EXPECT_LT(t.time_fused, t.time_unfused);
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

  auto* operand0 = module->entry_computation()->root_instruction()->operand(0);
  auto* operand1 = module->entry_computation()->root_instruction()->operand(1);

  // DUS scales with the size of the updates, so these two fusions should have
  // the same cost.
  auto t1 = EstimateRunTimes(operand0);
  auto t2 = EstimateRunTimes(operand1);
  EXPECT_NEAR(absl::ToInt64Microseconds(t1.time_unfused),
              absl::ToInt64Microseconds(t2.time_unfused), 10);
}

TEST_F(GpuPerformanceModelTest, EqualCostBeforeAndAfterFusion) {
  absl::string_view hlo_string = R"(
HloModule m

f1 {
  p0 = f32[4194304] parameter(0)
  p1 = f32[4194304] parameter(1)
  ROOT tmp_3 = f32[4194304] multiply(f32[4194304] p0, f32[4194304] p1)
}

e1 {
  p0 = f32[4194304] parameter(0)
  p1 = f32[4194304] parameter(1)

  f.1 = f32[4194304] fusion(f32[4194304] p0, f32[4194304] p1), kind=kLoop, calls=f1
  ROOT r.1 = f32[4194304] tanh(f32[4194304] f.1)
}

f2 {
  p0 = f32[4194304] parameter(0)
  p1 = f32[4194304] parameter(1)
  mul = f32[4194304] multiply(f32[4194304] p0, f32[4194304] p1)
  ROOT res = f32[4194304] tanh(f32[4194304] mul)
}

ENTRY e2 {
  p0 = f32[4194304] parameter(0)
  p1 = f32[4194304] parameter(1)

  ROOT f.2 = f32[4194304] fusion(f32[4194304] p0, f32[4194304] p1), kind=kLoop, calls=f2
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* computation_without_fusion =
      module->GetComputationWithName("e1");
  ASSERT_IS_OK(computation_without_fusion->Accept(&analysis_));
  HloInstruction* consumer = computation_without_fusion->root_instruction();
  const HloInstruction* producer = consumer->operand(0);

  auto t1 = EstimateRunTimes(producer, {consumer});

  HloComputation* computation_with_fusion =
      module->GetComputationWithName("e2");
  ASSERT_IS_OK(computation_with_fusion->Accept(&analysis_));
  HloInstruction* root_with_fusion =
      computation_with_fusion->root_instruction();

  auto t2 = EstimateRunTimes(root_with_fusion);
  EXPECT_EQ(t1.time_fused, t2.time_unfused);
}

TEST_F(GpuPerformanceModelTest, DoNotFuseDivideIntoSmallReduce) {
  // Fusing this divide is not supported by reduce epilogue fusion.
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

ENTRY fusion {
  c = f32[] constant(0)
  p0 = f32[3072] parameter(0)
  p1 = f32[] parameter(1)
  reduce = f32[] reduce(p0, c), dimensions={0}, to_apply=add
  ROOT divide = f32[] divide(reduce, p1)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer =
      module->entry_computation()->GetInstructionWithName("reduce");
  std::vector<HloInstruction*> consumers{
      module->entry_computation()->GetInstructionWithName("divide")};

  auto t = EstimateRunTimes(producer, consumers);
  EXPECT_LT(t.time_unfused, t.time_fused);
}

TEST_F(GpuPerformanceModelTest, PreferFusingExpensiveInstructionsIntoProducer) {
  // All things being equal, prefer fusing instructions into their producer,
  // since this avoids potentially expensive recomputations when memory and
  // compute aren't perfectly overlapping.
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_computation.0 {
  p0 = f32[4,8,8] parameter(0)
  bc = f32[1,4,1424,8,8] broadcast(p0), dimensions={1,3,4}
  p1 = f32[1,4,1424,8,8] parameter(1)
  ROOT sub = f32[1,4,1424,8,8] subtract(bc, p1)
}

fused_computation.1 {
  p0 = f32[1,4,1424,8,8] parameter(0)
  bc = f32[4,1424,8,8] bitcast(p0)
  c0 = f32[] constant(0)
  ROOT reduce = f32[4,8,8] reduce(bc, c0), to_apply=add, dimensions={1}
}

ENTRY fusion {
  p0 = f32[4,8,8] parameter(0)
  p1 = f32[1,4,1424,8,8] parameter(1)
  fusion.0 = f32[1,4,1424,8,8] fusion(p0, p1), kind=kLoop, calls=fused_computation.0
  exp = f32[1,4,1424,8,8] exponential(fusion.0)
  ROOT fusion.1 = f32[4,8,8] fusion(exp), kind=kInput, calls=fused_computation.1
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* fusion_0 =
      module->entry_computation()->GetInstructionWithName("fusion.0");
  auto* exp = module->entry_computation()->GetInstructionWithName("exp");
  auto exp_consumer_runtimes = EstimateRunTimes(fusion_0, {exp});
  auto exp_producer_runtimes = EstimateRunTimes(exp, exp->users());

  auto exp_consumer_priority =
      exp_consumer_runtimes.time_unfused - exp_consumer_runtimes.time_fused;
  auto exp_producer_priority =
      exp_producer_runtimes.time_unfused - exp_producer_runtimes.time_fused;

  EXPECT_LT(exp_producer_priority, exp_consumer_priority);
}

TEST_F(GpuPerformanceModelTest, DontFuseExpensiveElementwiseIntoSmallReduce) {
  constexpr absl::string_view kHlo = R"(
HloModule testmodule

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_computation.0 {
  p0 = f32[4,256,32] parameter(0)
  tanh = f32[4,256,32] tanh(p0)
  c1 = f32[] constant(72)
  broadcast = f32[4,256, 32] broadcast(c1), dimensions={}
  ROOT mul = f32[4,256,32] multiply(tanh, broadcast)
}

ENTRY fusion {
  p0 = f32[4,256,32] parameter(0)
  fusion = f32[4,256,32] fusion(p0), kind=kLoop, calls=fused_computation.0
  c0 = f32[] constant(0)
  ROOT reduce = f32[4,32] reduce(fusion, c0), to_apply=add, dimensions={1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* fusion = module->entry_computation()->GetInstructionWithName("fusion");
  auto* reduce = module->entry_computation()->GetInstructionWithName("reduce");

  auto t = EstimateRunTimes(fusion, {reduce});

  EXPECT_LT(t.time_unfused, t.time_fused);
}

TEST_F(GpuPerformanceModelTest,
       EstimateRunTimeForFusion_InfiniteProducer_ReturnsInfinite) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule testmodule

ENTRY fusion {
  p0 = f32[32] parameter(0)
  exp = f32[32] exponential(p0)
  ROOT add = f32[32] add(p0, exp)
})"));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer = module->entry_computation()->GetInstructionWithName("exp");
  auto* consumer = module->entry_computation()->GetInstructionWithName("add");

  auto config = GpuPerformanceModelOptions::Default(
      &fusion_analysis_cache_, &gpu_performance_model_cache_);

  auto producer_runtime = EstimateRunTimeData::Infinite();
  gpu_performance_model_cache_.Set(*producer, producer_runtime);

  auto consumer_runtime = GpuPerformanceModel::EstimateRunTimeForInstruction(
      consumer, device_info_, &analysis_, config);

  auto result = GpuPerformanceModel::EstimateRunTimeForFusion(
      producer, consumer, producer_runtime, consumer_runtime, device_info_,
      &analysis_, config);

  EXPECT_EQ(result, absl::InfiniteDuration());
}

TEST_F(GpuPerformanceModelTest,
       EstimateRunTimeForFusion_InfiniteConsumer_ReturnsInfinite) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule testmodule

ENTRY fusion {
  p0 = f32[32] parameter(0)
  exp = f32[32] exponential(p0)
  ROOT add = f32[32] add(p0, exp)
})"));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer = module->entry_computation()->GetInstructionWithName("exp");
  auto* consumer = module->entry_computation()->GetInstructionWithName("add");

  auto config = GpuPerformanceModelOptions::Default(
      &fusion_analysis_cache_, &gpu_performance_model_cache_);

  auto producer_runtime = GpuPerformanceModel::EstimateRunTimeForInstruction(
      producer, device_info_, &analysis_, config);

  auto consumer_runtime = EstimateRunTimeData::Infinite();
  gpu_performance_model_cache_.Set(*producer, consumer_runtime);

  auto result = GpuPerformanceModel::EstimateRunTimeForFusion(
      producer, consumer, producer_runtime, consumer_runtime, device_info_,
      &analysis_, config);

  EXPECT_EQ(result, absl::InfiniteDuration());
}

TEST_F(GpuPerformanceModelTest,
       EstimateRunTimeForFusion_MultiOutputWrite_ReturnsCorrectTime) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule m

fused_power {
  constant.1 = f32[] constant(2)
  broadcast = f32[4,28672,28672] broadcast(constant.1), dimensions={}
  iota = s32[28672,28672] iota(), iota_dimension=0
  iota.1 = s32[28672,28672] iota(), iota_dimension=1
  compare = pred[28672,28672] compare(iota, iota.1), direction=GE
  broadcast.1 = pred[4,28672,28672] broadcast(compare), dimensions={1,2}
  param_0.3 = f32[4,28672,28672] parameter(0)
  constant.2 = f32[] constant(-1e+30)
  broadcast.2 = f32[4,28672,28672] broadcast(constant.2), dimensions={}
  select = f32[4,28672,28672] select(broadcast.1, param_0.3, broadcast.2)
  param_1.2 = f32[4,28672] parameter(1)
  broadcast.3 = f32[4,28672,28672] broadcast(param_1.2), dimensions={0,1}
  subtract = f32[4,28672,28672] subtract(select, broadcast.3)
  ROOT power = f32[4,28672,28672] power(broadcast, subtract)
}

region.1 {
  param_0.1 = f32[] parameter(0)
  param_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(param_0.1, param_1.1)
}

fused_reduce {
  param_0.2 = f32[4,28672,28672] parameter(0)
  bitcast = f32[4,28672,128,224] bitcast(param_0.2)
  constant = f32[] constant(0)
  ROOT reduce = f32[4,28672,128] reduce(bitcast, constant), dimensions={3}, to_apply=region.1
}

ENTRY entry_computation.1 {
  param_1.3 = f32[4,28672,28672] parameter(1)
  param_0.4 = f32[4,28672] parameter(0)
  loop_power_fusion = f32[4,28672,28672] fusion(param_1.3, param_0.4), kind=kLoop, calls=fused_power
  input_reduce_fusion = f32[4,28672,128] fusion(loop_power_fusion), kind=kInput, calls=fused_reduce
  ROOT tuple = (f32[4,28672,28672], f32[4,28672,128]) tuple(loop_power_fusion, input_reduce_fusion)
})"));
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  auto* producer =
      module->entry_computation()->GetInstructionWithName("loop_power_fusion");
  auto* consumer = module->entry_computation()->GetInstructionWithName(
      "input_reduce_fusion");

  auto t = GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion(
      producer, consumer, device_info_, &analysis_);
  EXPECT_NEAR(absl::ToInt64Milliseconds(t.time_unfused), 162, 1);
  EXPECT_NEAR(absl::ToInt64Milliseconds(t.time_fused), 145, 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
