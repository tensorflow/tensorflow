/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_hlo_schedule.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {
namespace gpu {

using ::testing::ElementsAre;
using ::tsl::testing::StatusIs;

class GpuHloScheduleTest : public HloTestBase {
 protected:
  using HloVec = std::vector<HloInstruction*>;

  // Pre-canned shapes.
  Shape f32_2x2_ = ShapeUtil::MakeShape(F32, {2, 2});

  SequentialHloOrdering BuildHloOrdering(HloModule* module) {
    Backend& test_backend = backend();
    const se::DeviceDescription& gpu_device_info =
        test_backend.default_stream_executor()->GetDeviceDescription();
    TF_CHECK_OK(ScheduleGpuModule(module, /*pointer_size=*/8, gpu_device_info)
                    .status());
    return SequentialHloOrdering{module->schedule()};
  }

  struct TestConfig {
    bool enable_latency_hiding_scheduler = false;
    bool enable_pipelined_p2p = false;
    std::string fdo_profile = "";
  };

  HloModuleConfig GetModuleConfig(const TestConfig test_config) {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_latency_hiding_scheduler(
        test_config.enable_latency_hiding_scheduler);
    debug_options.set_xla_gpu_enable_pipelined_p2p(
        test_config.enable_pipelined_p2p);
    config.set_debug_options(debug_options);
    config.set_fdo_profile(test_config.fdo_profile);
    return config;
  }

  std::unique_ptr<HloModule> CreateNewVerifiedModule(
      bool enable_latency_hiding_scheduler = false) {
    TestConfig test_config;
    test_config.enable_latency_hiding_scheduler =
        enable_latency_hiding_scheduler;
    return std::make_unique<HloModule>("test_module",
                                       GetModuleConfig(test_config));
  }

  static bool HasValidFingerprint(HloModule* module) {
    // Verify that the fingerprint of HLO prior to LHS is present.
    const FrontendAttributes& attrs = module->frontend_attributes();
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

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
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

TEST_F(GpuHloScheduleTest,
       ScheduleGpuModuleWithMemorySchedulerReturnsPeakMemoryBytes) {
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(kHloText, GetModuleConfig(test_config)));
  int64_t pointer_size =
      dynamic_cast<GpuCompiler*>(backend().compiler())->GetPointerSize();
  int64_t peak_memory_bytes = -1;
  TF_ASSERT_OK_AND_ASSIGN(auto schedule,
                          ScheduleGpuModuleWithMemoryScheduler(
                              module.get(), pointer_size, &peak_memory_bytes));
  EXPECT_GT(peak_memory_bytes, 0);
}

TEST_F(GpuHloScheduleTest, LHSCostModelCostlyAR) {
  const char* hlo_text = R"(
  HloModule AsyncAR
  apply_op {
    x = bf16[] parameter(0)
    y = bf16[] parameter(1)
    ROOT apply_op = bf16[] add(x, y)
  }

  ENTRY ar {
    p0 = bf16[32505856] parameter(0)
    p1 = f32[32, 32] parameter(1)
    p2 = f32[32, 32] parameter(2)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(dot0, p2), custom_call_target="__cublas$gemm"
    dot2 = f32[32,32]{1,0} custom-call(dot1, p2), custom_call_target="__cublas$gemm"
    dot3 = f32[32,32]{1,0} custom-call(dot2, p2), custom_call_target="__cublas$gemm"
    dot4 = f32[32,32]{1,0} custom-call(dot3, p2), custom_call_target="__cublas$gemm"
    dot5 = f32[32,32]{1,0} custom-call(dot4, p2), custom_call_target="__cublas$gemm"
    dot6 = f32[32,32]{1,0} custom-call(dot5, p2), custom_call_target="__cublas$gemm"

    ar-start = bf16[32505856] all-reduce-start(p0), to_apply=apply_op
    ar-done = bf16[32505856] all-reduce-done(ar-start)

    ROOT t = (bf16[32505856], f32[32,32]) tuple(ar-done, dot6)
  })";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

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

  EXPECT_EQ(count_between_pairs.size(), 1);
  // We pack in 7 medium cost operations into the costly AR.
  // By default we pack in at most 5.
  EXPECT_EQ(count_between_pairs[0], 7);
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

TEST_F(GpuHloScheduleTest, ProfileGuidedCostModel) {
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

    // Independent compute
    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    // 2 Independent collectives.
    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    ROOT t = (f32[32], f32[32], f32[32,32]) tuple(ar-done, ar-done1, add0)
  })";

  struct SubTest {
    std::string profile;
    std::string target_start, target_done;
  };
  std::vector<SubTest> subtests;

  // Subtest 1: execute with text profile. ar-start/done having long latency,
  // whereas ar-start1/ar-done1 having short latency. So we expect all compute
  // to be scheduled between ar-start/ar-done
  const std::string ar_long_latency_proto_text = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
    costs { name: "dot1" cost_us: 100.0 }
    costs { name: "add0" cost_us: 10.0 }
    costs { name: "ar-start" cost_us: 1000.0 }
    costs { name: "ar-start1" cost_us: 10.0 }
  )pb";
  subtests.push_back({ar_long_latency_proto_text, "ar-start", "ar-done"});

  // Subtest 1: execute with binary profile. ar-start1/done having long latency,
  // whereas ar-start/ar-done having short latency. So we expect all compute
  // to be scheduled between ar-start1/ar-done1
  const std::string ar1_long_latency_proto_text = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
    costs { name: "dot1" cost_us: 100.0 }
    costs { name: "add0" cost_us: 10.0 }
    costs { name: "ar-start" cost_us: 10.0 }
    costs { name: "ar-start1" cost_us: 1000.0 }
  )pb";
  tensorflow::profiler::ProfiledInstructionsProto profile;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      ar1_long_latency_proto_text, &profile));
  std::string ar1_long_latency_proto_binary = profile.SerializeAsString();
  subtests.push_back({profile.SerializeAsString(), "ar-start1", "ar-done1"});

  for (const SubTest& subtest : subtests) {
    TestConfig test_config;
    test_config.enable_latency_hiding_scheduler = true;
    test_config.fdo_profile = subtest.profile;
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
    SequentialHloOrdering order = BuildHloOrdering(module.get());

    HloComputation* entry = module->entry_computation();

    // We expect all the math instructions between ar-start/ar-done
    bool between_target_collective_pair = false;
    for (const HloInstruction* inst :
         order.SequentialOrder(*entry)->instructions()) {
      if (inst->name() == subtest.target_start) {
        between_target_collective_pair = true;
      } else if (inst->name() == subtest.target_done) {
        between_target_collective_pair = false;
      } else if (inst->opcode() == HloOpcode::kDot ||
                 inst->opcode() == HloOpcode::kAdd) {
        EXPECT_TRUE(between_target_collective_pair);
      }
    }
  }
}

TEST_F(GpuHloScheduleTest, ProfileGuidedCostModelFailsWithIncompleteProfile) {
  const absl::string_view kHloString = R"(
  HloModule m

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32,32] parameter(1)
    p2 = f32[32,32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    ROOT t = (f32[32],f32[32],f32[32,32]) tuple(ar-done, ar-done1, add0)
  })";

  // Profile string, cost does not matter.
  const absl::string_view kProfile = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
    costs { name: "add0" cost_us: 10.0 }
    costs { name: "ar-start" cost_us: 1000.0 }
  )pb";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.fdo_profile = kProfile;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(kHloString, GetModuleConfig(test_config)));

  HloModuleConfig config(module->config());
  DebugOptions dboptions(config.debug_options());
  dboptions.set_xla_gpu_pgle_accuracy_checker(
      DebugOptions::PGLE_STRICTNESS_LEVEL_ERROR);
  config.set_debug_options(dboptions);
  module->set_config(config);

  // `dot1` and `ar-start1` are missing from the profile.
  EXPECT_THAT(ScheduleGpuModule(
                  module.get(), /*pointer_size=*/8,
                  backend().default_stream_executor()->GetDeviceDescription())
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(
    GpuHloScheduleTest,
    ProfileGuidedCostModelDoesNotFailWithIncompleteProfileIfAccuracyCheckerIsDisabled) {  // NOLINT(whitespace/line_length)
  const absl::string_view kHloString = R"(
  HloModule m

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32,32] parameter(1)
    p2 = f32[32,32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    ROOT t = (f32[32],f32[32],f32[32,32]) tuple(ar-done, ar-done1, add0)
  })";

  // Profile string, cost does not matter.
  const absl::string_view kProfile = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
    costs { name: "add0" cost_us: 10.0 }
    costs { name: "ar-start" cost_us: 1000.0 }
  )pb";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.fdo_profile = kProfile;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(kHloString, GetModuleConfig(test_config)));

  // `dot1` and `ar-start1` are missing from the profile but we disable the
  // pass.
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "pgle-accuracy-checker");
  TF_EXPECT_OK(ScheduleGpuModule(
                   module.get(), /*pointer_size=*/8,
                   backend().default_stream_executor()->GetDeviceDescription())
                   .status());
}

TEST_F(GpuHloScheduleTest, ProfileGuidedCostModelWithRematData) {
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

    // Independent compute
    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    // Independent collectives.
    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ar-start1 = f32[32] all-reduce-start(p3), to_apply=apply_op
    ar-done1 = f32[32] all-reduce-done(ar-start1)

    ROOT t = (f32[32], f32[32], f32[32,32]) tuple(ar-done, ar-done1, add0)
  })";

  // Costs of "ar-start" and "ar-start.remat100" should be averaged out and
  // used as cost for "ar-start".
  const std::string ar_long_latency_proto_text = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
    costs { name: "dot1" cost_us: 100.0 }
    costs { name: "add0" cost_us: 10.0 }
    costs { name: "ar-start" cost_us: 1.0 }
    costs { name: "ar-start1" cost_us: 1.0 }
    costs { name: "ar-start.remat100" cost_us: 2000.0 }
  )pb";
  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.fdo_profile = ar_long_latency_proto_text;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

  HloComputation* entry = module->entry_computation();

  // We expect all the math instructions between ar-start/ar-done
  bool between_target_collective_pair = false;
  for (const HloInstruction* inst :
       order.SequentialOrder(*entry)->instructions()) {
    if (inst->name() == "ar-start") {
      between_target_collective_pair = true;
    } else if (inst->name() == "ar-done") {
      between_target_collective_pair = false;
    } else if (inst->opcode() == HloOpcode::kDot ||
               inst->opcode() == HloOpcode::kAdd) {
      EXPECT_TRUE(between_target_collective_pair);
    }
  }
}

// Checks that the Send and Recv sequence created by the CollectivePermute
// decomposer is properly scheduled:
//  recv
//  send
//  recv-done
//  computation
//  send-done
TEST_F(GpuHloScheduleTest, LHSSendRecv) {
  const char* hlo_text = R"(
  HloModule test
  while_cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(25)
    ROOT cond_result = pred[] compare(count, ub), direction=LT
  }

  while_body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    send-data = get-tuple-element(%param), index=1

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    }
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}

    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}
    while_init = (u32[], f32[1, 1024, 1024]) tuple(c0, init)
    while_result = (u32[], f32[1, 1024, 1024]) while(while_init),
      body=while_body, condition=while_cond
    ROOT entry_result = f32[1, 1024, 1024] get-tuple-element(while_result),
      index=1
  }
  )";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.enable_pipelined_p2p = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());
  HloComputation* while_body = module->GetComputationWithName("while_body");
  const std::vector<HloInstruction*>& instruction_sequence =
      order.SequentialOrder(*while_body)->instructions();
  auto get_index = [&](absl::string_view hlo_name) {
    return absl::c_find_if(instruction_sequence,
                           [hlo_name](HloInstruction* instruction) {
                             return instruction->name() == hlo_name;
                           }) -
           instruction_sequence.begin();
  };

  EXPECT_LT(get_index("recv"), get_index("send"));
  EXPECT_LT(get_index("send"), get_index("recv-done"));
  EXPECT_GE(get_index("send-done") - get_index("recv-done"), 8);
  EXPECT_LT(abs(get_index("send-done") - get_index("result")), 2);
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

// Checks that the two pairs of (Recv, RecvDone) and (Send, SendDone) do not
// interleave.
TEST_F(GpuHloScheduleTest, LHSSendRecvPairs2) {
  const char* hlo_text = R"(
  HloModule test
  while_cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(25)
    ROOT cond_result = pred[] compare(count, ub), direction=LT
  }

  while_body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    send-data = get-tuple-element(%param), index=1

    after-all-0 = token[] after-all()
    recv-0 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all-0), channel_id=1,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    }
    send-0 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all-0),
      channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    }
    recv-done-0 = (f32[1, 1024, 1024], token[]) recv-done(recv-0), channel_id=1
    send-done-0 = token[] send-done(send-0), channel_id=1
    recv-data-0 = f32[1, 1024, 1024] get-tuple-element(recv-done-0), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    bc1 = f32[1, 1024, 1024] broadcast(conv), dimensions={}

    after-all-1 = token[] after-all()
    recv-1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all-1), channel_id=2,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{1, 0}}"
    }
    send-1 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all-1),
      channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{1, 0}}"
    }
    recv-done-1 = (f32[1, 1024, 1024], token[]) recv-done(recv-1), channel_id=2
    send-done-1 = token[] send-done(send-1), channel_id=2
    recv-data-1 = f32[1, 1024, 1024] get-tuple-element(recv-done-1), index=0

    add2 = f32[1, 1024, 1024] add(recv-data-0, bc1)
    add = f32[1, 1024, 1024] add(recv-data-1, add2)

    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new_count, add)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}
    while_init = (u32[], f32[1, 1024, 1024]) tuple(c0, init)
    while_result = (u32[], f32[1, 1024, 1024]) while(while_init),
      body=while_body, condition=while_cond
    ROOT entry_result = f32[1, 1024, 1024] get-tuple-element(while_result), index=1
  }
  )";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.enable_pipelined_p2p = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());
  HloComputation* while_body = module->GetComputationWithName("while_body");
  const std::vector<HloInstruction*>& instruction_sequence =
      order.SequentialOrder(*while_body)->instructions();
  auto get_index = [&](absl::string_view hlo_name) {
    return absl::c_find_if(instruction_sequence,
                           [hlo_name](HloInstruction* instruction) {
                             return instruction->name() == hlo_name;
                           }) -
           instruction_sequence.begin();
  };

  EXPECT_TRUE(HasValidFingerprint(module.get()));

  EXPECT_LT(get_index("recv-1"), get_index("send-1"));
  EXPECT_LT(get_index("send-1"), get_index("recv-done-1"));
  EXPECT_GT(get_index("send-done-1"), get_index("send-1"));

  EXPECT_LT(get_index("send-done-1"), get_index("recv-0"));
  EXPECT_LT(abs(get_index("send-done-0") - get_index("result")), 2);
}

// Checks that asynchronous AllReduce is scheduled to not interleave with the
// Send and Recv sequence.
TEST_F(GpuHloScheduleTest, LHSSendRecvAllReduce) {
  const char* hlo_text = R"(
  HloModule test
  add (x: f32[], y: f32[]) -> f32[] {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(f32[] x, f32[] y)
  }

  while_cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(25)
    ROOT cond_result = pred[] compare(count, ub), direction=LT
  }

  while_body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    send-data = get-tuple-element(%param), index=1

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}}"
    }
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}

    all-reduce-start = f32[1, 1024, 1024] all-reduce-start(f32[1, 1024, 1024] p),
      replica_groups={{0,1}}, to_apply=add,  backend_config={"collective_backend_config":{"is_sync":false}}
    all-reduce-done = f32[1, 1024, 1024] all-reduce-done(f32[1, 1024, 1024] all-reduce-start)
    new-data = f32[1, 1024, 1024] add(s, all-reduce-done)
    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new_count, new-data)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}
    while_init = (u32[], f32[1, 1024, 1024]) tuple(c0, init)
    while_result = (u32[], f32[1, 1024, 1024]) while(while_init),
      body=while_body, condition=while_cond
    ROOT entry_result = f32[1, 1024, 1024] get-tuple-element(while_result), index=1
  }
  )";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.enable_pipelined_p2p = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());
  HloComputation* while_body = module->GetComputationWithName("while_body");
  const std::vector<HloInstruction*>& instruction_sequence =
      order.SequentialOrder(*while_body)->instructions();
  auto get_index = [&](absl::string_view hlo_name) {
    return absl::c_find_if(instruction_sequence,
                           [hlo_name](HloInstruction* instruction) {
                             return instruction->name() == hlo_name;
                           }) -
           instruction_sequence.begin();
  };

  EXPECT_LT(get_index("recv"), get_index("send"));
  EXPECT_LT(get_index("send"), get_index("recv-done"));
  EXPECT_GE(get_index("send-done") - get_index("recv-done"), 3);
  EXPECT_TRUE(get_index("send-done") < get_index("all-reduce-start") ||
              get_index("recv") > get_index("all-reduce-start"));
  EXPECT_TRUE(HasValidFingerprint(module.get()));
}

// Checks that with the dependence added by the gpu-hlo-scheduler, the
// pipelined one Send-Recv group is scheduled correctly.
TEST_F(GpuHloScheduleTest, LHSSendRecvPipelined1) {
  const char* hlo_text = R"(
  HloModule test

  while_cond {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while_body {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    send-data = f32[1, 1024, 1024] add(c, s)

    after-all.1 = token[] after-all()
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.1),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.1 = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.1 = token[] send-done(send.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done.1, send-done.1)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.2 = token[] after-all()
    recv.2 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send.2 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.2 = (f32[1,1024,1024], token[]) recv-done(recv.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.2 = token[] send-done(send.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    while-init =  (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(c0, recv-done.2, send-done.2)
    while-result = (u32[], (f32[1,1024,1024], token[]), token[])
      while(while-init),
      body=while_body, condition=while_cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.2.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result), index=1

    ROOT entry-result = f32[1, 1024, 1024] get-tuple-element(recv-done.2.q), index=0
  }
  )";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.enable_pipelined_p2p = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());
  const std::vector<HloInstruction*>& while_body =
      order.SequentialOrder(*module->GetComputationWithName("while_body"))
          ->instructions();
  const std::vector<HloInstruction*>& main =
      order.SequentialOrder(*module->GetComputationWithName("main"))
          ->instructions();
  auto get_index =
      [](absl::string_view hlo_name,
         const std::vector<HloInstruction*>& instruction_sequence) {
        return absl::c_find_if(instruction_sequence,
                               [hlo_name](HloInstruction* instruction) {
                                 return instruction->name() == hlo_name;
                               }) -
               instruction_sequence.begin();
      };
  EXPECT_TRUE(HasValidFingerprint(module.get()));

  // The pipelined Send-Recv in the main. A pipelined Recv is scheduled right
  // after its corresponding Send due to kForceEarly.
  EXPECT_EQ(get_index("recv.2", main) + 3, get_index("send.2", main));
  EXPECT_LT(get_index("send.2", main), get_index("recv-done.2", main));
  EXPECT_LT(get_index("recv-done.2", main), get_index("send-done.2", main));
  EXPECT_LT(get_index("send-done.2", main), get_index("while-result", main));

  // The pipelined Send-Recv in the while-body. A pipelined Recv is scheduled
  // right after its corresponding Send due to kForceEarly.
  EXPECT_EQ(get_index("recv.1", while_body) + 1,
            get_index("send.1", while_body));
  EXPECT_LT(get_index("send.1", while_body),
            get_index("recv-done.1", while_body));
  EXPECT_LT(get_index("recv-done.1", while_body),
            get_index("send-done.1", while_body));
}

// Checks that with the dependence added by the gpu-hlo-scheduler, the
// pipelined two Send-Recv groups are scheduled correctly.
TEST_F(GpuHloScheduleTest, LHSSendRecvPipelined2) {
  const char* hlo_text = R"(
  HloModule test

  while_cond {
    param = (u32[], (f32[1,1024,1024],  token[]), token[],
      (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while_body {
    param = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.0.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    recv-data.0 = f32[1, 1024, 1024] get-tuple-element(recv-done.0.q), index=0
    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=3
    recv-data.1 = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[1, 1024, 1024] broadcast(compare0), dimensions={}
    recv-data = f32[1, 1024, 1024] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    send-data = f32[1, 1024, 1024] add(c, s)

    after-all.0 = token[] after-all()
    send.0 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.0),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv.0 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.0 = (f32[1,1024,1024], token[]) recv-done(recv.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.0 = token[] send-done(send.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.1 = token[] after-all()
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.1),
      channel_id=2, frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
      }
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
        _xla_send_recv_pipeline="1"
      }
    recv-done.1 = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.1 = token[] send-done(send.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done.0, send-done.0, recv-done.1, send-done.1)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.2 = token[] after-all()
    recv.2 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.2), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{3,0}}",
       _xla_send_recv_pipeline="0"
    }
    send.2 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.2), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{3,0}}",
       _xla_send_recv_pipeline="0"
    }
    recv-done.2 = (f32[1,1024,1024], token[]) recv-done(recv.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.2 = token[] send-done(send.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.3 = token[] after-all()
    recv.3 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.3), channel_id=2,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
    }
    send.3 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.3), channel_id=2,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
    }
    recv-done.3 = (f32[1,1024,1024], token[]) recv-done(recv.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.3 = token[] send-done(send.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    while-init =  (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) tuple(c0, recv-done.2, send-done.2, recv-done.3, send-done.3)
    while-result = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) while(while-init),
      body=while_body, condition=while_cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.2.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result), index=1
    recv-data.2 = f32[1, 1024, 1024] get-tuple-element(recv-done.2.q), index=0
    recv-done.3.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result), index=3
    recv-data.3 = f32[1, 1024, 1024] get-tuple-element(recv-done.3.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[1, 1024, 1024] broadcast(compare0), dimensions={}
    ROOT entry-result = f32[1, 1024, 1024] select(compare, recv-data.2, recv-data.3)
  }
  )";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.enable_pipelined_p2p = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());
  const std::vector<HloInstruction*>& while_body =
      order.SequentialOrder(*module->GetComputationWithName("while_body"))
          ->instructions();
  const std::vector<HloInstruction*>& main =
      order.SequentialOrder(*module->GetComputationWithName("main"))
          ->instructions();
  auto get_index =
      [](absl::string_view hlo_name,
         const std::vector<HloInstruction*>& instruction_sequence) {
        return absl::c_find_if(instruction_sequence,
                               [hlo_name](HloInstruction* instruction) {
                                 return instruction->name() == hlo_name;
                               }) -
               instruction_sequence.begin();
      };

  EXPECT_TRUE(HasValidFingerprint(module.get()));
  // The pipelined Send-Recv in the main. A pipelined Recv is scheduled right
  // after its corresponding Send due to kForceEarly.
  EXPECT_EQ(get_index("recv.2", main) + 3, get_index("send.2", main));
  EXPECT_LT(get_index("send.2", main), get_index("recv.3", main));
  EXPECT_EQ(get_index("recv.3", main) + 1, get_index("send.3", main));
  EXPECT_LT(get_index("send.3", main), get_index("recv-done.2", main));
  EXPECT_LT(get_index("recv-done.2", main), get_index("recv-done.3", main));
  EXPECT_LT(get_index("recv-done.3", main), get_index("send-done.2", main));
  EXPECT_LT(get_index("send-done.2", main), get_index("send-done.3", main));
  EXPECT_LT(get_index("send-done.3", main), get_index("while-result", main));

  // The pipelined Send-Recv in the while-body. A pipelined Recv is scheduled
  // right after its corresponding Send due to kForceEarly.
  EXPECT_EQ(get_index("recv.0", while_body) + 1,
            get_index("send.0", while_body));
  EXPECT_LT(get_index("send.0", while_body), get_index("recv.1", while_body));
  EXPECT_EQ(get_index("recv.1", while_body) + 1,
            get_index("send.1", while_body));
  EXPECT_LT(get_index("send.1", while_body),
            get_index("recv-done.0", while_body));
  EXPECT_LT(get_index("recv-done.0", while_body),
            get_index("recv-done.1", while_body));
  EXPECT_LT(get_index("recv-done.1", while_body),
            get_index("send-done.0", while_body));
  EXPECT_LT(get_index("send-done.0", while_body),
            get_index("send-done.1", while_body));
}

TEST_F(GpuHloScheduleTest, SkipAlreadyScheduled) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m, is_scheduled=true

fused_computation {
  param_0 = f32[1024,1024]{1,0} parameter(0)
  ROOT exponential.1 = f32[1024,1024]{1,0} exponential(param_0)
}

fused_computation.1 {
  param_0.1 = f32[1024,1024]{1,0} parameter(0)
  ROOT negate.1 = f32[1024,1024]{1,0} negate(param_0.1)
}

ENTRY e {
  p = f32[1024,1024]{1,0} parameter(0)
  wrapped_negate = f32[1024,1024]{1,0} fusion(p), kind=kLoop, calls=fused_computation.1
  wrapped_exponential = f32[1024,1024]{1,0} fusion(p), kind=kLoop, calls=fused_computation
  ROOT t = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(wrapped_exponential, wrapped_negate)
})")
                    .value();
  TF_CHECK_OK(ScheduleGpuModule(
                  module.get(), /*pointer_size=*/8,
                  backend().default_stream_executor()->GetDeviceDescription())
                  .status());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY
// CHECK: wrapped_negate = f32[1024,1024]{1,0}
// CHECK: wrapped_exponential = f32[1024,1024]{1,0}
)"));
}

TEST_F(GpuHloScheduleTest, ProfileGuidedCostModelWithForceEarliestSchedule) {
  const char* hlo_text = R"(
  HloModule AsyncAR
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY main {
    p0 = f32[32] parameter(0)
    p1 = f32[32, 32] parameter(1)
    p2 = f32[32, 32] parameter(2)
    p3 = f32[32] parameter(3)

    // Independent compute
    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm", backend_config={"force_earliest_schedule":true}
    dot1 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    add0 = f32[32,32] add(dot0, dot1)

    // 2 Independent collectives.
    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    ROOT t = (f32[32], f32[32,32]) tuple(ar-done, add0)
  })";

  const std::string ar_long_latency_proto_text = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
    costs { name: "dot1" cost_us: 100.0 }
    costs { name: "add0" cost_us: 10.0 }
    costs { name: "ar-start" cost_us: 1000.0 }
  )pb";

  tensorflow::profiler::ProfiledInstructionsProto profile;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      ar_long_latency_proto_text, &profile));
  std::string ar_long_latency_proto_binary = profile.SerializeAsString();

  // Post processing should work even with GpuAsyncTrackerBase.
  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  test_config.fdo_profile = ar_long_latency_proto_binary;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

  const std::vector<HloInstruction*>& main =
      order.SequentialOrder(*module->GetComputationWithName("main"))
          ->instructions();
  auto get_index =
      [](absl::string_view hlo_name,
         const std::vector<HloInstruction*>& instruction_sequence) {
        return absl::c_find_if(instruction_sequence,
                               [hlo_name](HloInstruction* instruction) {
                                 return instruction->name() == hlo_name;
                               }) -
               instruction_sequence.begin();
      };
  // Using the profile, LHS should schedule all computes between ar pair,
  // but since dot0 is marked as force delay, it should be scheduled
  // before ar-start now.
  EXPECT_LT(get_index("dot0", main), get_index("ar-start", main));
  // Also verify that dot1 is scheduled between ar start and ar done.
  EXPECT_GT(get_index("dot1", main), get_index("ar-start", main));
  EXPECT_LT(get_index("dot1", main), get_index("ar-done", main));
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
          /*device_list=*/CollectiveDeviceList(), /*constrain_layout=*/false,
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

TEST_F(GpuHloScheduleTest, LHSResourceModel) {
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

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig(test_config)));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

  uint32_t in_flight = 0;
  uint32_t max_in_flight = 0;
  for (const HloInstruction* inst :
       order.SequentialOrder(*module->entry_computation())->instructions()) {
    if (hlo_query::IsAsyncCollectiveStartOp(inst)) {
      in_flight++;
      max_in_flight = std::max(max_in_flight, in_flight);
    } else if (hlo_query::IsAsyncCollectiveDoneOp(inst)) {
      in_flight--;
    }
  }

  EXPECT_EQ(max_in_flight, 1);
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
    ag-start = (f32[32], f32[64]) all-gather-start(p1), dimensions={0}, backend_config="{\"collective_backend_config\":{\"is_sync\":true}}"
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

  const std::vector<absl::string_view> expected_sequence = {
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

TEST_F(GpuHloScheduleTest, AsyncOps) {
  const char* hlo_text = R"(
  HloModule m

  op1 {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  op2 {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  ENTRY main {
    p0 = f32[2,2] parameter(0)
    // The `async-start` blocks should be moved up, and the `async-done` blocks
    // should be moved down.
    acc1_start = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
        kind=kLoop, calls=op1
    acc1_done = f32[2,2] fusion-done(acc1_start)
    acc2_start = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
        kind=kLoop, calls=op2
    acc2_done = f32[2,2] fusion-done(acc2_start)
    ROOT done = f32[2,2] add(acc1_done, acc2_done)
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(hlo_text, HloModuleConfig{}));
  SequentialHloOrdering order = BuildHloOrdering(module.get());

  std::vector<HloOpcode> opcodes;
  for (HloInstruction* instruction :
       order.SequentialOrder(*module->entry_computation())->instructions()) {
    opcodes.push_back(instruction->opcode());
  }
  EXPECT_THAT(opcodes,
              ElementsAre(HloOpcode::kParameter, HloOpcode::kAsyncStart,
                          HloOpcode::kAsyncStart, HloOpcode::kAsyncDone,
                          HloOpcode::kAsyncDone, HloOpcode::kAdd));
}

// This test verifies that the latency hiding scheduler overlaps host memory
// offloading (copy-start/copy-done) with computation.
TEST_F(GpuHloScheduleTest, CopyStartDoneScheduled) {
  constexpr absl::string_view kHloCopyStartDone = R"(
    HloModule offloading
      ENTRY main {
      param.0 = f32[512,1024]{1,0} parameter(0)
      tanh.14 = f32[512,1024]{1,0} tanh(param.0)
      copy-start.1 = (f32[512,1024]{1,0:S(5)}, f32[512,1024]{1,0}, u32[]) copy-start(param.0)
      copy-done.1 = f32[512,1024]{1,0:S(5)} copy-done(copy-start.1)
      copy-start.3 = (f32[512,1024]{1,0}, f32[512,1024]{1,0:S(5)}, u32[]) copy-start(copy-done.1)
      copy-done.3 = f32[512,1024]{1,0} copy-done(copy-start.3)
      ROOT add.0 = f32[512,1024]{1,0} add(copy-done.3, tanh.14)
  })";

  TestConfig test_config;
  test_config.enable_latency_hiding_scheduler = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kHloCopyStartDone,
                                                GetModuleConfig(test_config)));
  TF_CHECK_OK(ScheduleGpuModule(
                  module.get(), /*pointer_size=*/8,
                  backend().default_stream_executor()->GetDeviceDescription())
                  .status());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY
// CHECK: copy-start.3 = (f32[512,1024]{1,0}, f32[512,1024]{1,0:S(5)}, u32[]) copy-start
// CHECK: tanh.14 = f32[512,1024]{1,0} tanh
// CHECK: copy-done.3 = f32[512,1024]{1,0} copy-done
)"));
}

}  // namespace gpu
}  // namespace xla
