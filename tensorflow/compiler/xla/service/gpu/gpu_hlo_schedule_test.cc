/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {
namespace gpu {

class GpuHloScheduleTest : public HloTestBase {
 protected:
  using HloVec = std::vector<HloInstruction*>;

  // Pre-canned shapes.
  Shape f32_2x2_ = ShapeUtil::MakeShape(F32, {2, 2});

  SequentialHloOrdering BuildHloOrdering(HloModule* module) {
    Backend& test_backend = backend();
    const GpuDeviceInfo gpu_device_info =
        GetGpuDeviceInfo(test_backend.default_stream_executor());
    TF_CHECK_OK(ScheduleGpuModule(module, /*pointer_size=*/8, gpu_device_info));
    return SequentialHloOrdering{module->schedule()};
  }

  HloModuleConfig GetModuleConfig(bool enable_latency_hiding_scheduler,
                                  bool enable_gpu_async_tracker = false) {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_latency_hiding_scheduler(
        enable_latency_hiding_scheduler);
    debug_options.set_xla_gpu_lhs_enable_gpu_async_tracker(
        enable_gpu_async_tracker);
    config.set_debug_options(debug_options);
    return config;
  }

  std::unique_ptr<HloModule> CreateNewVerifiedModule(
      bool enable_latency_hiding_scheduler = false) {
    return std::make_unique<HloModule>(
        "test_module", GetModuleConfig(enable_latency_hiding_scheduler));
  }

  static bool HasValidFingerprint(HloModule* module) {
    // Verify that the fingerprint of HLO prior to LHS is present.
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    const FrontendAttributes& attrs = root->frontend_attributes();
    auto it = attrs.map().find(kFingerprintBeforeLHS);

    // The fingerprint is 128 bits stored as a hex string (128/4 hex digits).
    return it != attrs.map().end() && it->second.size() == 128 / 4;
  }
};

// Test of a single stream, where data dependencies fully determine the
// execution order.
TEST_F(GpuHloScheduleTest, SequentialMatMul) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* dot1 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, x, y));
  HloInstruction* dot2 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, dot1, z));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build(dot2));

  SequentialHloOrdering order = BuildHloOrdering(module.get());
  EXPECT_TRUE(order.ExecutesBefore(y, x));
  EXPECT_TRUE(order.ExecutesBefore(y, dot1));
  EXPECT_TRUE(order.ExecutesBefore(z, dot1));
  EXPECT_TRUE(order.ExecutesBefore(z, dot2));
  EXPECT_TRUE(order.ExecutesBefore(dot1, dot2));
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

// Test of a single stream, where data dependencies do not fully determine the
// execution order, but the stream assignment does.
TEST_F(GpuHloScheduleTest, SequentialAdd) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, x, y));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, y, z));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, add2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build(add3));

  SequentialHloOrdering order = BuildHloOrdering(module.get());
  EXPECT_TRUE(order.ExecutesBefore(y, x));
  EXPECT_TRUE(order.ExecutesBefore(y, add1));
  EXPECT_TRUE(order.ExecutesBefore(z, add1));
  EXPECT_TRUE(order.ExecutesBefore(z, add2));
  EXPECT_TRUE(order.ExecutesBefore(add1, add2));
  EXPECT_TRUE(order.ExecutesBefore(add2, add3));
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

TEST_F(GpuHloScheduleTest, AsyncCustomCall) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, x, y));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add0, y));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, z));
  // Create nonblocking_call(add0).
  HloInstruction* nonblocking_call =
      builder.AddInstruction(HloInstruction::CreateCustomCall(
          f32_2x2_, {add0},
          /*custom_call_target=*/"nonblocking-call-start",
          /*opaque=*/""));
  static_cast<HloCustomCallInstruction*>(nonblocking_call)
      ->set_custom_call_schedule(SCHEDULE_EARLIEST);
  // In addition, add control_dependency: add1->nonblocking_call.
  TF_CHECK_OK(add1->AddControlDependencyTo(nonblocking_call));
  // Blocking call, which only add4 depends on.
  HloInstruction* blocking_call =
      builder.AddInstruction(HloInstruction::CreateCustomCall(
          f32_2x2_, {nonblocking_call},
          /*custom_call_target=*/"blocking-call-done",
          /*opaque=*/""));
  static_cast<HloCustomCallInstruction*>(blocking_call)
      ->set_custom_call_schedule(SCHEDULE_LATEST);
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, add2));
  HloInstruction* add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_2x2_, HloOpcode::kAdd, add3, blocking_call));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build(add4));

  SequentialHloOrdering order = BuildHloOrdering(module.get());
  VLOG(2) << order.ToString();

  // Order constrained by data dependency.
  EXPECT_TRUE(order.ExecutesBefore(add0, nonblocking_call));
  // Order constrained by control dependency.
  EXPECT_TRUE(order.ExecutesBefore(add1, nonblocking_call));
  // Test that nonblocking_call is scheduled before add2, so that we know
  // EARLIEST is in effect.
  EXPECT_TRUE(order.ExecutesBefore(nonblocking_call, add2));
  EXPECT_TRUE(order.ExecutesBefore(nonblocking_call, add3));
  EXPECT_TRUE(order.ExecutesBefore(nonblocking_call, add4));

  // Test that blocking_call is scheduled after add3, so that we know
  // LATEST is in effect.
  EXPECT_TRUE(order.ExecutesBefore(add3, blocking_call));
  EXPECT_TRUE(order.ExecutesBefore(blocking_call, add4));
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

TEST_F(GpuHloScheduleTest, AsyncCollectivePermute) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();

  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, x, y));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add0, y));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, z));

  Shape u32_scalar = ShapeUtil::MakeShape(U32, {});

  Shape collective_permute_start_shape =
      ShapeUtil::MakeTupleShape({f32_2x2_, f32_2x2_});
  HloInstruction* collective_permute_start =
      builder.AddInstruction(HloInstruction::CreateCollectivePermuteStart(
          collective_permute_start_shape, add0,
          /*source_target_pairs=*/{{0, 1}}, /*channel_id=*/std::nullopt));
  // In addition, add control_dependency: add1->nonblocking_call.
  TF_CHECK_OK(add1->AddControlDependencyTo(collective_permute_start));
  // Blocking call, which only add4 depends on.
  HloInstruction* collective_permute_done = builder.AddInstruction(
      HloInstruction::CreateUnary(f32_2x2_, HloOpcode::kCollectivePermuteDone,
                                  collective_permute_start));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, add2));
  HloInstruction* add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_2x2_, HloOpcode::kAdd, add3, collective_permute_done));

  module->AddEntryComputation(builder.Build(add4));

  SequentialHloOrdering order = BuildHloOrdering(module.get());
  VLOG(2) << order.ToString();

  // Order constrained by data dependency.
  EXPECT_TRUE(order.ExecutesBefore(add0, collective_permute_start));
  // Order constrained by control dependency.
  EXPECT_TRUE(order.ExecutesBefore(add1, collective_permute_start));
  // Test that all_reduce_start is scheduled before add2.
  EXPECT_TRUE(order.ExecutesBefore(collective_permute_start, add2));
  EXPECT_TRUE(order.ExecutesBefore(collective_permute_start, add3));
  EXPECT_TRUE(order.ExecutesBefore(collective_permute_start, add4));

  // Test that all_reduce_done is scheduled after add3.
  EXPECT_TRUE(order.ExecutesBefore(add3, collective_permute_done));
  EXPECT_TRUE(order.ExecutesBefore(collective_permute_done, add4));
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

TEST_F(GpuHloScheduleTest, LHSCostModel) {
  const char* hlo_text = R"(
  HloModule AsyncAR
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32, 32] parameter(1)
    p2 = f32[32, 32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(dot0, p2), custom_call_target="__cublas$gemm"
    dot2 = f32[32,32]{1,0} custom-call(dot1, p2), custom_call_target="__cublas$gemm"
    dot3 = f32[32,32]{1,0} custom-call(dot2, p2), custom_call_target="__cublas$gemm"
    dot4 = f32[32,32]{1,0} custom-call(dot3, p2), custom_call_target="__cublas$gemm"
    dot5 = f32[32,32]{1,0} custom-call(dot4, p2), custom_call_target="__cublas$gemm"
    dot6 = f32[32,32]{1,0} custom-call(dot5, p2), custom_call_target="__cublas$gemm"

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    add0 = f32[32,32] add(dot0, dot1)
    add1 = f32[32,32] add(add0, dot2)
    add2 = f32[32,32] add(add1, dot3)
    add3 = f32[32,32] add(add2, dot4)
    add4 = f32[32,32] add(add3, dot5)
    add5 = f32[32,32] add(add4, dot6)

    ROOT t = (f32[32], f32[32], f32[32,32]) tuple(ar-done, ar-done1, add5)
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          hlo_text, GetModuleConfig(/*enable_latency_hiding_scheduler=*/true)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

  // With a better cost model, the latency hiding scheduler should distribute
  // the dots between both ar-start/done pairs. With a Nop cost model, they will
  // be clustered between only one of the pairs.
  HloComputation* entry = module->entry_computation();
  std::vector<int64_t> count_between_pairs;
  bool in_between = false;
  for (const HloInstruction* inst :
       order.SequentialOrder(*entry)->instructions()) {
    if (inst->opcode() == HloOpcode::kAllReduceStart) {
      in_between = true;
      count_between_pairs.push_back(0);
    } else if (inst->opcode() == HloOpcode::kAllReduceDone) {
      in_between = false;
    } else if (in_between && inst->opcode() == HloOpcode::kCustomCall) {
      count_between_pairs.back()++;
    }
  }

  EXPECT_EQ(count_between_pairs.size(), 2);
  EXPECT_GT(count_between_pairs[0], 0);
  EXPECT_GT(count_between_pairs[1], 0);
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

class GpuHloScheduleParameterizedTest
    : public GpuHloScheduleTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(GpuHloScheduleParameterizedTest, AsyncAllReduce) {
  // All-reduce reduction computation.
  HloComputation::Builder reduction_builder("add");
  HloInstruction* x0 =
      reduction_builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeScalarShape(F32),
          /*name=*/"x"));
  HloInstruction* y0 =
      reduction_builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/1, ShapeUtil::MakeScalarShape(F32),
          /*name=*/"y"));
  HloInstruction* add =
      reduction_builder.AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeScalarShape(F32), HloOpcode::kAdd, x0, y0));

  const bool use_latency_hiding_scheduler = GetParam();
  std::unique_ptr<HloModule> module =
      CreateNewVerifiedModule(use_latency_hiding_scheduler);
  HloComputation* reduction_computation =
      module->AddEmbeddedComputation(reduction_builder.Build(add));

  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, x, y));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add0, y));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, z));

  Shape all_reduce_start_shape =
      ShapeUtil::MakeTupleShape({f32_2x2_, f32_2x2_});
  HloInstruction* all_reduce_start =
      builder.AddInstruction(HloInstruction::CreateAllReduceStart(
          all_reduce_start_shape, {add0}, reduction_computation,
          /*replica_groups=*/{}, /*constrain_layout=*/false,
          /*channel_id=*/std::nullopt, /*use_global_device_ids=*/true));
  // In addition, add control_dependency: add1->nonblocking_call.
  TF_CHECK_OK(add1->AddControlDependencyTo(all_reduce_start));
  // Blocking call, which only add4 depends on.
  HloInstruction* all_reduce_done =
      builder.AddInstruction(HloInstruction::CreateUnary(
          f32_2x2_, HloOpcode::kAllReduceDone, all_reduce_start));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, add2));
  HloInstruction* add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_2x2_, HloOpcode::kAdd, add3, all_reduce_done));

  module->AddEntryComputation(builder.Build(add4));

  SequentialHloOrdering order = BuildHloOrdering(module.get());
  VLOG(2) << order.ToString();

  // Order constrained by data dependency.
  EXPECT_TRUE(order.ExecutesBefore(add0, all_reduce_start));
  // Order constrained by control dependency.
  EXPECT_TRUE(order.ExecutesBefore(add1, all_reduce_start));
  // Test that all_reduce_start is scheduled before add2.
  EXPECT_TRUE(order.ExecutesBefore(all_reduce_start, add2));
  EXPECT_TRUE(order.ExecutesBefore(all_reduce_start, add3));
  EXPECT_TRUE(order.ExecutesBefore(all_reduce_start, add4));

  // Test that all_reduce_done is scheduled after add3.
  EXPECT_TRUE(order.ExecutesBefore(add3, all_reduce_done));
  EXPECT_TRUE(order.ExecutesBefore(all_reduce_done, add4));
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

TEST_P(GpuHloScheduleParameterizedTest, LHSResourceModel) {
  const char* hlo_text = R"(
  HloModule AsyncModule
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32, 32] parameter(1)
    p2 = f32[32, 32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(dot0, p2), custom_call_target="__cublas$gemm"
    dot2 = f32[32,32]{1,0} custom-call(dot1, p2), custom_call_target="__cublas$gemm"
    dot3 = f32[32,32]{1,0} custom-call(dot2, p2), custom_call_target="__cublas$gemm"
    dot4 = f32[32,32]{1,0} custom-call(dot3, p2), custom_call_target="__cublas$gemm"
    dot5 = f32[32,32]{1,0} custom-call(dot4, p2), custom_call_target="__cublas$gemm"
    dot6 = f32[32,32]{1,0} custom-call(dot5, p2), custom_call_target="__cublas$gemm"

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    %ag-start = (f32[32], f32[64]) all-gather-start(p3), dimensions={0}
    %ag-done = f32[64] all-gather-done(%ag-start)

    add0 = f32[32,32] add(dot0, dot1)
    add1 = f32[32,32] add(add0, dot2)
    add2 = f32[32,32] add(add1, dot3)
    add3 = f32[32,32] add(add2, dot4)
    add4 = f32[32,32] add(add3, dot5)
    add5 = f32[32,32] add(add4, dot6)

    ROOT t = (f32[32], f32[64], f32[32,32]) tuple(ar-done, %ag-done, add5)
  })";

  const bool enable_gpu_async_tracker = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          hlo_text,
          GetModuleConfig(
              /*enable_latency_hiding_scheduler=*/true,
              /*enable_gpu_async_tracker=*/enable_gpu_async_tracker)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

  // Count the number of collectives in flight. Without gpu async tracker, we
  // will incorrectly have 2 in-flight (as base async tracker assumes each
  // collective can be scheduled independently as they use different resource
  // types), but with gpu async tracker we will have 1.
  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;
  for (const HloInstruction* inst :
       order.SequentialOrder(*module->entry_computation())->instructions()) {
    HloOpcode op = inst->opcode();
    if (hlo_query::IsAsyncCollectiveStartOp(op)) {
      in_flight++;
      max_in_flight = std::max(max_in_flight, in_flight);
    } else if (hlo_query::IsAsyncCollectiveDoneOp(op)) {
      in_flight--;
    }
  }

  const uint32_t expected_max_in_flight = enable_gpu_async_tracker ? 1 : 2;
  EXPECT_EQ(expected_max_in_flight, max_in_flight);
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

INSTANTIATE_TEST_SUITE_P(GpuHloScheduleParameterizedTest,
                         GpuHloScheduleParameterizedTest, ::testing::Bool());

using GpuHloSchedulePostProcessTest = HloTestBase;

TEST_F(GpuHloSchedulePostProcessTest, PostProcessAsyncCollectives) {
  const char* hlo_text = R"(
  HloModule AsyncModule, is_scheduled=true
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32] parameter(1)

    // This is async by default, so we expect the start/done to be moved.
    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    add0 = f32[32] add(p0, p0)
    ar-done = f32[32] all-reduce-done(ar-start)

    // This will be sync, so we expect the start/done to be moved next to each
    // other.
    ag-start = (f32[32], f32[64]) all-gather-start(p1), dimensions={0}, backend_config="{\"is_sync\":true}"
    add1 = f32[32] add(p1, p1)
    ag-done = f32[64] all-gather-done(ag-start)

    add2 = f32[32] add(add0, add1)
    add3 = f32[32] add(add2, ar-done)
    ROOT result = (f32[32], f32[64]) tuple(add3, ag-done)
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_text, /*replica_count=*/2));

  const HloInstructionSequence& input =
      module->schedule().sequence(module->entry_computation());
  HloInstructionSequence result = PostProcessSchedule(input);

  const std::vector<std::string_view> expected_sequence = {
      "p0",
      "ar-start",  // ar-start is async, should be scheduled as early as
                   // possible.
      "p1", "add0", "add1",
      "ag-start",  // ag-start is sync, so its scheduled right before its done.
      "ag-done", "add2",
      "ar-done",  // ar-done is async, should be scheduled as late as possible.
      "add3", "result"};

  ASSERT_EQ(expected_sequence.size(), result.size());
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_EQ(expected_sequence[i], result.instructions()[i]->name());
  }
}

}  // namespace gpu
}  // namespace xla
