/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/collective_pipeliner.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/service/collective_pipeliner_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/legalize_scheduling_annotations.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::_;
using ::tsl::testing::IsOkAndHolds;
namespace op = xla::testing::opcode_matchers;

class CollectivePipelinerTest : public HloHardwareIndependentTestBase {
 public:
  CollectivePipelinerTest() {
    const int64_t kNumReplicas = 4;
    const int64_t kNumPartitions = 2;
    config_ = GetModuleConfigForTest(/*replica_count=*/kNumReplicas,
                                     /*num_partitions=*/kNumPartitions);
  }

  static bool IsAllGatherExplicitPipeliningAnnotation(
      const HloInstruction* instr,
      collective_pipeliner_utils::PipeliningDirection direction) {
    std::optional<AnnotationIterationId> iteration_id =
        GetSchedulingAnnotationIterationId(instr).value();
    return IsAllGather(instr) && iteration_id &&
           IsIterationIdConstentWithPipeliningDirection(*iteration_id,
                                                        direction);
  }

 protected:
  static const HloPredicate IsAllGather;
  HloModuleConfig config_;
};

const HloPredicate CollectivePipelinerTest::IsAllGather =
    HloPredicateIsOp<HloOpcode::kAllGather>;

absl::StatusOr<bool> RunOptimizer(
    HloModule* module, bool last_run, int64_t level_to_operate_on = 0,
    bool pipeline_use_tree = false, bool process_different_sized_ops = true,
    collective_pipeliner_utils::PipeliningDirection direction =
        collective_pipeliner_utils::PipeliningDirection::kForward,
    HloPredicate should_process = HloPredicateIsOp<HloOpcode::kAllReduce>,
    HloPredicate acceptable_formatting = HloPredicateTrue,
    HloPredicate reuse_pipelined_op_buffer = HloPredicateTrue,
    HloPredicate should_allow_loop_variant_parameter_in_chain =
        HloPredicateFalse,
    CollectivePipeliner::HloPostprocessor postprocess_backward_peeled = {},
    CollectivePipeliner::HloPostprocessor postprocess_backward_rotated = {},
    CollectivePipeliner::HloPostprocessor postprocess_backward_peeled_trailing =
        {},
    bool should_add_loop_invariant_op_in_chain = false,
    int64_t collective_size_threshold_to_stop_sinking = INT64_MAX) {
  CollectivePipeliner::Config config = {
      /*level_to_operate_on=*/level_to_operate_on,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/last_run,
      /*pipeline_use_tree=*/pipeline_use_tree,
      /*process_different_sized_ops=*/process_different_sized_ops,
      /*direction=*/
      direction,
      /*should_process=*/should_process,
      /*acceptable_formatting=*/acceptable_formatting,
      /*reuse_pipelined_op_buffer=*/reuse_pipelined_op_buffer,
      should_allow_loop_variant_parameter_in_chain,
      /*should_allow_control_dependencies=*/false, postprocess_backward_peeled,
      postprocess_backward_rotated, postprocess_backward_peeled_trailing,
      should_add_loop_invariant_op_in_chain,
      /*postprocess_pipelined_ops=*/{},
      collective_size_threshold_to_stop_sinking};
  HloPassPipeline pass("optimizer");
  pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  pass.AddPass<CollectivePipeliner>(config);
  pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  return pass.Run(module);
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOne) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(_, op::AllReduce(), _, _, _));
  const HloInstruction* sliced = root->operand(1)->operand(0);
  EXPECT_EQ(sliced->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* index = sliced->operand(1);
  EXPECT_EQ(index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(index->tuple_index(), 3);
  const HloInstruction* while_inst = index->operand(0);
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_root =
      while_inst->while_body()->root_instruction();
  EXPECT_EQ(while_root->opcode(), HloOpcode::kTuple);
  const HloInstruction* dyn_upd = while_root->operand(1);
  EXPECT_EQ(dyn_upd->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* dyn_upd2 = dyn_upd->operand(0);
  EXPECT_EQ(dyn_upd2->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* prev_ar = dyn_upd2->operand(1);
  EXPECT_EQ(prev_ar->opcode(), HloOpcode::kAllReduce);
  const HloInstruction* dyn_slice_top = prev_ar->operand(0);
  EXPECT_EQ(dyn_slice_top->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* get_tuple_value = dyn_slice_top->operand(0);
  const HloInstruction* get_tuple_index = dyn_slice_top->operand(1);
  EXPECT_EQ(get_tuple_value->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_value->tuple_index(), 1);
  EXPECT_EQ(get_tuple_index->tuple_index(), 3);
}

TEST_F(CollectivePipelinerTest, MinimalCaseWithoutDefaultLayouts) {
  constexpr absl::string_view hlo_string = R"(
    HloModule module

    add {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT add = bf16[] add(lhs, rhs)
    }

    while_cond {
      param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      c3 = s32[] constant(3)
      ROOT cmp = pred[] compare(i, c3), direction=LT
    }

    while_body {
      param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dst_data = bf16[3,8,128] get-tuple-element(param), index=1
      src_data = bf16[3,8,128] get-tuple-element(param), index=2
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      i_plus_one = s32[] add(i, c1)
      src_data_slice = bf16[1,8,128] dynamic-slice(src_data, i, c0, c0),
          dynamic_slice_sizes={1,8,128}
      ar = bf16[1,8,128] all-reduce(src_data_slice), replica_groups={},
          to_apply=add, channel_id=1
      updated_buffer = bf16[3,8,128] dynamic-update-slice(dst_data, ar, i, c0,
          c0)
      ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(i_plus_one,
          updated_buffer, src_data)
    }

    ENTRY entry {
      c0 = s32[] constant(0)
      p0 = bf16[3,8,128] parameter(0)
      tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
      while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple),
          condition=while_cond, body=while_body
      ROOT dst_data = bf16[3,8,128] get-tuple-element(while), index=1
    }
  )";
  HloParserOptions parser_config;
  parser_config.set_fill_missing_layouts(false);
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(
                                           hlo_string, config_, parser_config));
  EXPECT_THAT(RunOptimizer(module.get(), /*last_run=*/true),
              IsOkAndHolds(true));

  XLA_VLOG_LINES(1, module->ToString());

  // Match root.
  const HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  EXPECT_THAT(
      root, op::DynamicUpdateSlice(
                op::GetTupleElement(while_instr, /*tuple_index=*/1),
                op::AllReduce(op::DynamicSlice(
                    op::GetTupleElement(while_instr, /*tuple_index=*/1),
                    op::GetTupleElement(while_instr, /*tuple_index=*/3), _, _)),
                op::GetTupleElement(while_instr, /*tuple_index=*/3), _, _));

  // Match while instruction.
  auto match_c0 = op::Constant(LiteralUtil::CreateR0<int32_t>(0));
  auto match_c1 = op::Constant(LiteralUtil::CreateR0<int32_t>(1));
  EXPECT_THAT(
      while_instr,
      op::While(op::Tuple(
          op::Add(op::GetTupleElement(op::Tuple(match_c0, _, _),
                                      /*tuple_index=*/0),
                  match_c1),
          op::DynamicUpdateSlice(
              op::GetTupleElement(op::Tuple(_, op::Parameter(0), _),
                                  /*tuple_index=*/1),
              op::DynamicSlice(
                  op::GetTupleElement(op::Tuple(_, _, op::Parameter(0)),
                                      /*tuple_index=*/2),
                  op::GetTupleElement(op::Tuple(match_c0, _, _),
                                      /*tuple_index=*/0),
                  _, _),
              op::GetTupleElement(op::Tuple(match_c0, _, _),
                                  /*tuple_index=*/0),
              _, _),
          op::GetTupleElement(op::Tuple(_, _, op::Parameter(0)),
                              /*tuple_index=*/2),
          op::GetTupleElement(op::Tuple(match_c0, _, _), /*tuple_index=*/0))));

  // Match while body.
  const HloInstruction* while_body_root =
      while_instr->while_body()->root_instruction();
  EXPECT_THAT(
      while_body_root,
      op::Tuple(
          op::Add(op::GetTupleElement(op::Parameter(0), /*tuple_index=*/0),
                  match_c1),
          op::DynamicUpdateSlice(
              op::DynamicUpdateSlice(
                  op::GetTupleElement(op::Parameter(0), /*tuple_index=*/1),
                  op::AllReduce(op::DynamicSlice(
                      op::GetTupleElement(op::Parameter(0), /*tuple_index=*/1),
                      op::GetTupleElement(op::Parameter(0), /*tuple_index=*/3),
                      _, _)),
                  op::GetTupleElement(op::Parameter(0), /*tuple_index=*/3), _,
                  _),
              op::DynamicSlice(
                  op::GetTupleElement(op::Parameter(0), /*tuple_index=*/2),
                  op::GetTupleElement(op::Parameter(0), /*tuple_index=*/0), _,
                  _),
              op::GetTupleElement(op::Parameter(0), /*tuple_index=*/0), _, _),
          _, _));
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneCollectivePermute) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(14)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  cp = bf16[3,8,128] collective-permute(get-tuple-element.5), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}},
                     frontend_attributes={_xla_send_recv_validation="{{0,6},{1,7},{2,8},{3,9},{4,10},{5,11},{6,12},{7,13}}"}
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(14)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(13)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(cp, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=2
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  config_.set_num_partitions(8);
  config_.set_replica_count(1);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: HloModule
    // CHECK: %while_body
    // CHECK:   %[[cp:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}_xla_send_recv_validation={{[{]}}{0,5},{0,6},{1,7},{2,8},{3,9},{4,10},{5,11},{6,12}{{[}]}}
    // CHECK:   %[[dus:.+]] = {{.+}} dynamic-slice({{.*}}%[[cp]], {{.*}})
    // CHECK:   %[[mul:.+]] = {{.+}} multiply({{.*}}%[[dus]], {{.*}}%[[dus]])
    // CHECK:   %[[dus2:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[mul]], {{.*}})
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[dus2]], {{.*}})
    // CHECK: }
    // CHECK: ENTRY %entry
    // CHECK:   %[[cp:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}{_xla_send_recv_validation={{[{]}}{0,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0}{{[}]}}
    // CHECK:   %[[ds:.+]] = {{.+}} dynamic-slice({{.*}}%[[cp]], {{.*}})
    // CHECK:   %[[mul:.+]] = {{.+}} multiply({{.*}}%[[ds]], {{.*}}%[[ds]])
    // CHECK:   %[[dus:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[mul]], {{.*}})
    // CHECK:   %[[tuple:.+]] = {{.+}} tuple({{.*}}%[[dus]], {{.*}})
    // CHECK:   {{.+}} = {{.+}} while({{.*}}%[[tuple]])
    // CHECK: }
  )"));
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneCollectivePermuteBackwardCycle) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(14)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  cp = bf16[3,8,128] collective-permute(get-tuple-element.5), channel_id=1, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}},
                     frontend_attributes={_xla_send_recv_validation="{{7,13},{6,12},{5,11},{4,10},{3,9},{2,8},{1,7},{0,6}}"}
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(14)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(13)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(cp, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=2
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  config_.set_num_partitions(8);
  config_.set_replica_count(1);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: HloModule
    // CHECK: %while_body
    // CHECK:   %[[cp:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}_xla_send_recv_validation={{[{]}}{6,12},{5,11},{4,10},{3,9},{2,8},{1,7},{0,6},{0,5}{{[}]}}
    // CHECK:   %[[dus:.+]] = {{.+}} dynamic-slice({{.*}}%[[cp]], {{.*}})
    // CHECK:   %[[mul:.+]] = {{.+}} multiply({{.*}}%[[dus]], {{.*}}%[[dus]])
    // CHECK:   %[[dus2:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[mul]], {{.*}})
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[dus2]], {{.*}})
    // CHECK: }
    // CHECK: ENTRY %entry
    // CHECK:   %[[cp:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}{_xla_send_recv_validation={{[{]}}{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{0,0}{{[}]}}
    // CHECK:   %[[ds:.+]] = {{.+}} dynamic-slice({{.*}}%[[cp]], {{.*}})
    // CHECK:   %[[mul:.+]] = {{.+}} multiply({{.*}}%[[ds]], {{.*}}%[[ds]])
    // CHECK:   %[[dus:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[mul]], {{.*}})
    // CHECK:   %[[tuple:.+]] = {{.+}} tuple({{.*}}%[[dus]], {{.*}})
    // CHECK:   {{.+}} = {{.+}} while({{.*}}%[[tuple]])
    // CHECK: }
  )"));
}

TEST_F(CollectivePipelinerTest, UpdateSendRecvChannelIdForHostTransfers) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  after-all = after-all()
  send.88 = (s32[], u32[], token[]) send(
      add.232, after-all), channel_id=2, is_host_transfer=true
  send-done.88 = token[] send-done(send.88), channel_id=2, is_host_transfer=true
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* entry_comp = module->entry_computation();
  auto* unrolled_send_done = entry_comp->GetInstructionWithName("send-done.0");
  ASSERT_THAT(unrolled_send_done, ::testing::NotNull());
  auto* unrolled_send = unrolled_send_done->operand(0);
  auto channel_id = [](const HloInstruction* instr) {
    return DynCast<HloChannelInstruction>(instr)->channel_id();
  };
  EXPECT_EQ(channel_id(unrolled_send), channel_id(unrolled_send_done));
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneNoReuse) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true, 0, false, true,
                  collective_pipeliner_utils::PipeliningDirection::kForward,
                  HloPredicateIsOp<HloOpcode::kAllReduce>,
                  /*acceptable_formatting=*/
                  [](const HloInstruction* i) { return true; },
                  /*reuse_pipelined_op_buffer=*/
                  [](const HloInstruction* i) { return false; })
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  EXPECT_EQ(while_instr->shape().tuple_shapes().size(), 5);
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneNotFirstIdx) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[8,3,128], bf16[8,3,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[8,3,128], bf16[8,3,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[8,3,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[8,3,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[8,1,128] dynamic-slice(get-tuple-element.5, constant.2561, select.1348, constant.2561), dynamic_slice_sizes={8,1,128}
  mul = bf16[8,1,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[8,1,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[8,3,128] dynamic-update-slice(get-tuple-element.395, ar.1, constant.2561, select.1348, constant.2561)
  ROOT tuple = (s32[], bf16[8,3,128], bf16[8,3,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[8,3,128] parameter(0)
  tuple = (s32[], bf16[8,3,128], bf16[8,3,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[8,3,128], bf16[8,3,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[8,3,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(_, op::AllReduce(), _, _, _));
  const HloInstruction* sliced = root->operand(1)->operand(0);
  EXPECT_EQ(sliced->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* index = sliced->operand(2);
  EXPECT_EQ(index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(index->tuple_index(), 3);
  const HloInstruction* while_inst = index->operand(0);
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_root =
      while_inst->while_body()->root_instruction();
  EXPECT_EQ(while_root->opcode(), HloOpcode::kTuple);
  const HloInstruction* dyn_upd = while_root->operand(1);
  EXPECT_EQ(dyn_upd->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* dyn_upd2 = dyn_upd->operand(0);
  EXPECT_EQ(dyn_upd2->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* prev_ar = dyn_upd2->operand(1);
  EXPECT_EQ(prev_ar->opcode(), HloOpcode::kAllReduce);
  const HloInstruction* dyn_slice_top = prev_ar->operand(0);
  EXPECT_EQ(dyn_slice_top->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* get_tuple_value = dyn_slice_top->operand(0);
  const HloInstruction* get_tuple_index = dyn_slice_top->operand(2);
  EXPECT_EQ(get_tuple_value->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_value->tuple_index(), 1);
  EXPECT_EQ(get_tuple_index->tuple_index(), 3);
}

TEST_F(CollectivePipelinerTest, TransformIncrementByTwo) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(_, op::AllReduce(), _, _, _));
  const HloInstruction* sliced = root->operand(1)->operand(0);
  EXPECT_EQ(sliced->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* index = sliced->operand(1);
  EXPECT_EQ(index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(index->tuple_index(), 3);
  const HloInstruction* while_inst = index->operand(0);
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_root =
      while_inst->while_body()->root_instruction();
  EXPECT_EQ(while_root->opcode(), HloOpcode::kTuple);
  const HloInstruction* dyn_upd = while_root->operand(1);
  EXPECT_EQ(dyn_upd->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* dyn_upd2 = dyn_upd->operand(0);
  EXPECT_EQ(dyn_upd2->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* prev_ar = dyn_upd2->operand(1);
  EXPECT_EQ(prev_ar->opcode(), HloOpcode::kAllReduce);
  const HloInstruction* dyn_slice_top = prev_ar->operand(0);
  EXPECT_EQ(dyn_slice_top->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* get_tuple_value = dyn_slice_top->operand(0);
  const HloInstruction* get_tuple_index = dyn_slice_top->operand(1);
  EXPECT_EQ(get_tuple_value->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_value->tuple_index(), 1);
  EXPECT_EQ(get_tuple_index->tuple_index(), 3);
}

TEST_F(CollectivePipelinerTest, NoTransformCantProveIndexDoesntWrap) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(4)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-1)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, TransformNegativeIndexIterationToZero) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/false).value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _,
                        op::CustomCall(op::AllReduce(op::DynamicSlice(
                                           op::GetTupleElement(op::While()),
                                           op::GetTupleElement(),
                                           op::Constant(), op::Constant())),
                                       op::Constant()),
                        op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, EscapedInputNoTransform) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.911 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-slice.911, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  c1 = bf16[1,8,128] broadcast(cc), dimensions={}
  c2 = bf16[3,8,128] broadcast(cc), dimensions={}
  tuple = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) tuple(c0, p0, c1, c2)
  while = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true).value());
}

TEST_F(CollectivePipelinerTest, TransformWithAg) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,8,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _, op::AllGather(op::GetTupleElement(op::While())),
                        op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, TransformWithAgWithFormatting) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,9,128], bf16[3,9,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,9,128], bf16[3,9,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,9,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,9,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,9,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,9,128}
  mul = bf16[1,9,128] multiply(dynamic-slice.99, dynamic-slice.99)
  cpd = bf16[] constant(0)
  %pd = bf16[1,16,128] pad(mul, cpd), padding=0_0x0_7x0_0
  rs.1 = bf16[1,2,128] reduce-scatter(pd), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,16,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}
  slc = bf16[1,9,128] slice(ag.1), slice={[0:1], [0:9], [0:128]}
  dynamic-update-slice.35 = bf16[3,9,128] dynamic-update-slice(get-tuple-element.395, slc, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,9,128], bf16[3,9,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,9,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,9,128], bf16[3,9,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,9,128], bf16[3,9,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,9,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::DynamicUpdateSlice(
                  _, op::Slice(op::AllGather(op::GetTupleElement(op::While()))),
                  op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, TransformWithAgInsertCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  constant.2561 = s32[] constant(0)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, get-tuple-element.394, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,8,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, get-tuple-element.394, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-8)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/false, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  RunOptimizer(module.get(), /*last_run=*/true, 1).value();
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  // Matching the pattern we expect for the output of the loop when an
  // all-gather is pipelined through the loop. We dynamic-slice the stacked
  // data, perform the all-gather and then put it in the stacked data again.
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _, op::AllGather(op::GetTupleElement(op::While())),
                        op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, PushAgOver) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[3,8,128]{2,1,0})->bf16[3,8,128]{2,1,0}}

%add (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %add = bf16[] add(bf16[] %lhs, bf16[] %rhs)
}

%while_body.clone (loop_peel_param: (s32[], bf16[3,8,128], s32[])) -> (s32[], bf16[3,8,128], s32[]) {
  %loop_peel_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=0
  %constant.7 = s32[] constant(1)
  %add.4 = s32[] add(s32[] %get-tuple-element.2, s32[] %constant.7)
  %get-tuple-element.3 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=2
  %constant.12 = s64[] constant(1)
  %custom-call = s32[] custom-call(s32[] %get-tuple-element.4, s64[] %constant.12), custom_call_target="InsertedByPreviousStep"
  %constant.13 = s32[] constant(0)
  %constant.10 = s32[] constant(0)
  %dynamic-slice.2 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13), dynamic_slice_sizes={1,8,128}
  %ar.2 = bf16[1,1,128]{2,1,0} reduce-scatter(bf16[1,8,128]{2,1,0} %dynamic-slice.2), channel_id=2, replica_groups={}, to_apply=%add, dimensions={1}
  %ag.2 = bf16[1,8,128]{2,1,0} all-gather(bf16[1,1,128]{2,1,0} %ar.2), channel_id=32, replica_groups={}, dimensions={1}
  %dynamic-update-slice.2 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, bf16[1,8,128]{2,1,0} %ag.2, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13)
  %dynamic-slice.1 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %get-tuple-element.2, s32[] %constant.10, s32[] %constant.10), dynamic_slice_sizes={1,8,128}
  %mul.2 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.1, bf16[1,8,128]{2,1,0} %dynamic-slice.1)
  %constant.15 = s32[] constant(0)
  %dynamic-update-slice.4 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %dynamic-update-slice.2, bf16[1,8,128]{2,1,0} %mul.2, s32[] %get-tuple-element.2, s32[] %constant.15, s32[] %constant.15)
  ROOT %tuple.3 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.4, bf16[3,8,128]{2,1,0} %dynamic-update-slice.4, s32[] %get-tuple-element.2)
}

%while_cond.clone (loop_peel_cond_param: (s32[], bf16[3,8,128], s32[])) -> pred[] {
  %loop_peel_cond_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %gte.1 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_cond_param), index=0
  %constant.6 = s32[] constant(0)
  ROOT %cmp.1 = pred[] compare(s32[] %gte.1, s32[] %constant.6), direction=LT
}

ENTRY %entry (p0: bf16[3,8,128]) -> bf16[3,8,128] {
  %c0 = s32[] constant(-3)
  %p0 = bf16[3,8,128]{2,1,0} parameter(0)
  %tuple.1 = (s32[], bf16[3,8,128]{2,1,0}) tuple(s32[] %c0, bf16[3,8,128]{2,1,0} %p0)
  %get-tuple-element.0 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=0
  %constant.0 = s32[] constant(1)
  %constant.4 = s32[] constant(0)
  %add.1 = s32[] add(s32[] %get-tuple-element.0, s32[] %constant.0)
  %get-tuple-element.1 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=1
  %dynamic-slice.0 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4), dynamic_slice_sizes={1,8,128}
  %mul.1 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.0, bf16[1,8,128]{2,1,0} %dynamic-slice.0)
  %dynamic-update-slice.0 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, bf16[1,8,128]{2,1,0} %mul.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4)
  %tuple.4 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.1, bf16[3,8,128]{2,1,0} %dynamic-update-slice.0, s32[] %get-tuple-element.0)
  %while.1 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) while((s32[], bf16[3,8,128]{2,1,0}, s32[]) %tuple.4), condition=%while_cond.clone, body=%while_body.clone
  %get-tuple-element.6 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=1
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=2
  %constant.14 = s32[] constant(0)
  %dynamic-slice.3 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14), dynamic_slice_sizes={1,8,128}
  %ar.3 = bf16[1,8,128]{2,1,0} all-reduce(bf16[1,8,128]{2,1,0} %dynamic-slice.3), channel_id=3, replica_groups={}, to_apply=%add
  ROOT %dynamic-update-slice.3 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, bf16[1,8,128]{2,1,0} %ar.3, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14)
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 1,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  // Check that the all-gather can be pipelined after we had already a previous
  // round of pipelining performed previously for another op. (in this case
  // AllReduce).
  EXPECT_THAT(
      root,
      op::DynamicUpdateSlice(
          op::DynamicUpdateSlice(_, op::AllGather(), _, _, _),
          op::AllReduce(op::DynamicSlice(op::DynamicUpdateSlice(), _, _, _)),
          op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, NoPushAgOverBecauseDifferentSize) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[3,8,128]{2,1,0})->bf16[3,8,128]{2,1,0}}

%add (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %add = bf16[] add(bf16[] %lhs, bf16[] %rhs)
}

%while_body.clone (loop_peel_param: (s32[], bf16[3,8,128], s32[])) -> (s32[], bf16[3,8,128], s32[]) {
  %loop_peel_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=0
  %constant.7 = s32[] constant(1)
  %add.4 = s32[] add(s32[] %get-tuple-element.2, s32[] %constant.7)
  %get-tuple-element.3 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=2
  %constant.12 = s64[] constant(1)
  %custom-call = s32[] custom-call(s32[] %get-tuple-element.4, s64[] %constant.12), custom_call_target="InsertedByPreviousStep"
  %constant.13 = s32[] constant(0)
  %constant.10 = s32[] constant(0)
  %dynamic-slice.2 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13), dynamic_slice_sizes={1,8,128}
  %ar.2 = bf16[1,1,128]{2,1,0} reduce-scatter(bf16[1,8,128]{2,1,0} %dynamic-slice.2), channel_id=2, replica_groups={}, to_apply=%add, dimensions={1}
  %ag.2 = bf16[1,8,128]{2,1,0} all-gather(bf16[1,1,128]{2,1,0} %ar.2), channel_id=32, replica_groups={}, dimensions={1}
  %dynamic-update-slice.2 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, bf16[1,8,128]{2,1,0} %ag.2, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13)
  %dynamic-slice.1 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %get-tuple-element.2, s32[] %constant.10, s32[] %constant.10), dynamic_slice_sizes={1,8,128}
  %mul.2 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.1, bf16[1,8,128]{2,1,0} %dynamic-slice.1)
  %constant.15 = s32[] constant(0)
  %dynamic-update-slice.4 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %dynamic-update-slice.2, bf16[1,8,128]{2,1,0} %mul.2, s32[] %get-tuple-element.2, s32[] %constant.15, s32[] %constant.15)
  ROOT %tuple.3 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.4, bf16[3,8,128]{2,1,0} %dynamic-update-slice.4, s32[] %get-tuple-element.2)
}

%while_cond.clone (loop_peel_cond_param: (s32[], bf16[3,8,128], s32[])) -> pred[] {
  %loop_peel_cond_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %gte.1 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_cond_param), index=0
  %constant.6 = s32[] constant(0)
  ROOT %cmp.1 = pred[] compare(s32[] %gte.1, s32[] %constant.6), direction=LT
}

ENTRY %entry (p0: bf16[3,8,128]) -> bf16[3,8,128] {
  %c0 = s32[] constant(-3)
  %p0 = bf16[3,8,128]{2,1,0} parameter(0)
  %tuple.1 = (s32[], bf16[3,8,128]{2,1,0}) tuple(s32[] %c0, bf16[3,8,128]{2,1,0} %p0)
  %get-tuple-element.0 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=0
  %constant.0 = s32[] constant(1)
  %constant.4 = s32[] constant(0)
  %add.1 = s32[] add(s32[] %get-tuple-element.0, s32[] %constant.0)
  %get-tuple-element.1 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=1
  %dynamic-slice.0 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4), dynamic_slice_sizes={1,8,128}
  %mul.1 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.0, bf16[1,8,128]{2,1,0} %dynamic-slice.0)
  %dynamic-update-slice.0 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, bf16[1,8,128]{2,1,0} %mul.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4)
  %tuple.4 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.1, bf16[3,8,128]{2,1,0} %dynamic-update-slice.0, s32[] %get-tuple-element.0)
  %while.1 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) while((s32[], bf16[3,8,128]{2,1,0}, s32[]) %tuple.4), condition=%while_cond.clone, body=%while_body.clone
  %get-tuple-element.6 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=1
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=2
  %constant.14 = s32[] constant(0)
  %dynamic-slice.3 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14), dynamic_slice_sizes={1,8,128}
  %ar.3 = bf16[1,8,128]{2,1,0} all-reduce(bf16[1,8,128]{2,1,0} %dynamic-slice.3), channel_id=3, replica_groups={}, to_apply=%add
  ROOT %dynamic-update-slice.3 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, bf16[1,8,128]{2,1,0} %ar.3, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14)
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(module.get(), /*last_run=*/false, 1,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, TransformIncrementByTwoFormat) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,16,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,16,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,16,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,16,128] dynamic-slice(get-tuple-element.396, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,16,128}
  mul = bf16[1,16,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,16,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ds.1 = bf16[1,8,128] dynamic-slice(ar.1, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ds.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,16,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.396)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,16,128] parameter(0)
  c1 = bf16[] constant(0)
  b1 = bf16[3,8,128] broadcast(c1), dimensions={}
  tuple = (s32[], bf16[3,8,128], bf16[3,16,128]) tuple(c0, b1, p0)
  while = (s32[], bf16[3,8,128], bf16[3,16,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::DynamicUpdateSlice(
          _, op::DynamicSlice(op::AllReduce(op::GetTupleElement()), _, _, _), _,
          _, _));
}

TEST_F(CollectivePipelinerTest, TransformIncrementByTwoFormatTranspose) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,16,128], bf16[3,16,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,16,128], bf16[3,16,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,16,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,16,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,16,128] dynamic-slice(get-tuple-element.396, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,16,128}
  mul = bf16[1,16,128] multiply(dynamic-slice.99, dynamic-slice.99)
  reshape.1 = bf16[2,16,64] reshape(mul)
  ar.1 = bf16[2,16,64] all-reduce(reshape.1), replica_groups={}, to_apply=add, channel_id=1
  transpose.1 = bf16[64,2,16] transpose(ar.1), dimensions={2,0,1}
  reshape.2 = bf16[1,16,128] reshape(transpose.1)
  dynamic-update-slice.35 = bf16[3,16,128] dynamic-update-slice(get-tuple-element.395, reshape.2, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,16,128], bf16[3,16,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.396)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,16,128] parameter(0)
  c1 = bf16[] constant(0)
  b1 = bf16[3,16,128] broadcast(c1), dimensions={}
  tuple.1 = (s32[], bf16[3,16,128], bf16[3,16,128]) tuple(c0, b1, p0)
  while = (s32[], bf16[3,16,128], bf16[3,16,128]) while(tuple.1), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,16,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::DynamicUpdateSlice(
          _, op::Reshape(op::Transpose(op::AllReduce(op::GetTupleElement()))),
          _, _, _));
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneBackwards) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r), control-predecessors={constant.2559}
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k), control-predecessors={a}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  const int64_t while_count = absl::c_count_if(
      module->entry_computation()->instructions(),
      [](const HloInstruction* instruction) {
        return HloPredicateIsOp<HloOpcode::kWhile>(instruction);
      });
  EXPECT_EQ(while_count, 1);
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  const HloInstruction* tuple = while_instr->operand(0);
  EXPECT_TRUE(tuple->HasControlDependencies());
  EXPECT_EQ(tuple->control_predecessors().size(), 1);
  const HloInstruction* add_instr = tuple->control_predecessors()[0];
  EXPECT_EQ(add_instr->opcode(), HloOpcode::kAdd);
  const HloComputation* comp = while_instr->while_body();
  const HloInstruction* root_loop = comp->root_instruction();
  EXPECT_TRUE(root_loop->HasControlDependencies());
  EXPECT_EQ(root_loop->control_predecessors().size(), 1);
  const HloInstruction* add_instr_loop = root_loop->control_predecessors()[0];
  EXPECT_EQ(add_instr_loop->opcode(), HloOpcode::kAdd);
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneStartFromOneBackwards) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

while_cond {
  param = (s32[], bf16[5,8,128], bf16[5,1,2,128]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c4 = s32[] constant(4)
  ROOT cmp = pred[] compare(loop_index, c4), direction=LT
}

while_body {
  param = (s32[], bf16[5,8,128], bf16[5,1,2,128]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,8,128] get-tuple-element(param), index=1
  slice_input = bf16[5,1,2,128] get-tuple-element(param), index=2
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  c3 = s32[] constant(3)
  three_minus_loop_index = s32[] subtract(c3, loop_index)
  dynamic_slice = bf16[1,1,2,128] dynamic-slice(slice_input, three_minus_loop_index, c0, c0, c0), dynamic_slice_sizes={1,1,2,128}
  dynamic_slice_reshape = bf16[1,2,128] reshape(dynamic_slice)
  add = bf16[1,2,128] add(dynamic_slice_reshape, dynamic_slice_reshape), control-predecessors={c3}
  all_gather = bf16[1,8,128] all-gather(add), dimensions={1}, replica_groups={}
  updated_partial_output = bf16[5,8,128] dynamic-update-slice(partial_output, all_gather, three_minus_loop_index, c0, c0)
  ROOT tuple = (s32[], bf16[5,8,128], bf16[5,1,2,128]) tuple(next_loop_index, updated_partial_output, slice_input), control-predecessors={add}
}

ENTRY entry {
  c1 = s32[] constant(1)
  p0 = bf16[5,8,128] parameter(0)
  p1 = bf16[5,1,2,128] parameter(1)
  tuple = (s32[], bf16[5,8,128], bf16[5,1,2,128]) tuple(c1, p0, p1)
  while = (s32[], bf16[5,8,128], bf16[5,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  const HloComputation* comp = while_instr->while_body();
  const HloInstruction* root_loop = comp->root_instruction();

  const HloInstruction* shifted_loop_counter = root_loop->operand(4);
  EXPECT_EQ(shifted_loop_counter->opcode(), HloOpcode::kAdd);
  const HloInstruction* loop_increment = shifted_loop_counter->operand(1);
  EXPECT_EQ(loop_increment->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(loop_increment->literal().IsEqualAt({}, 1));
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsWithTwoDependentClones) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  // To be peeled.
  custom-call = bf16[1,2,128] custom-call(r), custom_call_target="MoveToDevice"
  a = bf16[1,2,128] add(custom-call, custom-call), control-predecessors={constant.2559}
  // To be peeled.
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k), control-predecessors={a}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto is_all_gather_or_offloading = [](const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kAllGather ||
           instruction->IsCustomCall(
               memory_annotations::kMoveToDeviceCustomCallTarget);
  };
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   is_all_gather_or_offloading)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  const int64_t while_count = absl::c_count_if(
      module->entry_computation()->instructions(),
      [](const HloInstruction* instruction) {
        return HloPredicateIsOp<HloOpcode::kWhile>(instruction);
      });
  EXPECT_EQ(while_count, 1);
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  const HloInstruction* tuple = while_instr->operand(0);
  EXPECT_TRUE(tuple->HasControlDependencies());
  EXPECT_EQ(tuple->control_predecessors().size(), 1);
  const HloInstruction* add_instr = tuple->control_predecessors()[0];
  EXPECT_EQ(add_instr->opcode(), HloOpcode::kAdd);
  const HloComputation* comp = while_instr->while_body();
  const HloInstruction* root_loop = comp->root_instruction();
  EXPECT_TRUE(root_loop->HasControlDependencies());
  EXPECT_EQ(root_loop->control_predecessors().size(), 1);
  const HloInstruction* add_instr_loop = root_loop->control_predecessors()[0];
  EXPECT_EQ(add_instr_loop->opcode(), HloOpcode::kAdd);

  EXPECT_NE(FindInstruction(module.get(), "custom-call.1"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "custom-call.2"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "ag.1"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "ag.2"), nullptr);
}

TEST_F(CollectivePipelinerTest, LoopVariantAppearingInRootTupleMultipleTimes) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], s32[], s32[]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], s32[], s32[]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  add.233 = s32[] add(add.232, constant.2557)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  // To be peeled.
  custom-call = bf16[1,2,128] custom-call(r), custom_call_target="MoveToDevice"
  a = bf16[1,2,128] add(custom-call, custom-call), control-predecessors={constant.2559}
  // To be peeled.
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], s32[], s32[]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k, add.233, add.233), control-predecessors={a}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  p2 = s32[] parameter(2)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], s32[], s32[]) tuple(c0, p0, p1, p2, p2)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128], s32[], s32[]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto is_all_gather_or_offloading = [](const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kAllGather ||
           instruction->IsCustomCall(
               memory_annotations::kMoveToDeviceCustomCallTarget);
  };
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   is_all_gather_or_offloading)
          .value());
}

TEST_F(CollectivePipelinerTest, TwoIterations) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(2)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  param3 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, get-tuple-element.394, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  // To be peeled.
  custom-call = bf16[1,2,128] custom-call(r), custom_call_target="MoveToDevice"
  a = bf16[1,2,128] add(custom-call, custom-call), control-predecessors={constant.2559}
  // To be peeled.
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, get-tuple-element.394, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, get-tuple-element.394, constant.2561, constant.2561)
  ar.2 = bf16[1,8,128] custom-call(ar.1), custom_call_target="MoveToHost"
  hmm = bf16[3,8,128] dynamic-update-slice(param3, ar.2, get-tuple-element.394, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k, hmm), control-predecessors={a}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  p2 = bf16[3,8,128] parameter(2)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) tuple(c0, p0, p1, p2)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto is_all_gather_or_offloading = [](const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kAllGather ||
           instruction->IsCustomCall(
               memory_annotations::kMoveToDeviceCustomCallTarget) ||
           instruction->IsCustomCall(
               memory_annotations::kMoveToHostCustomCallTarget);
  };
  bool changed =
      RunOptimizer(module.get(), /*last_run=*/true, /*level_to_operate_on=*/0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   is_all_gather_or_offloading)
          .value();
  ASSERT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  changed =
      RunOptimizer(module.get(), /*last_run=*/true, /*level_to_operate_on=*/0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   is_all_gather_or_offloading)
          .value();
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsCollectivePermute) {
  constexpr absl::string_view hlo_string = R"(
HloModule module
add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}
while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(14)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}
while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  cp = bf16[3,8,128] collective-permute(get-tuple-element.395), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}},
                     frontend_attributes={_xla_send_recv_validation="{{0,6},{1,7},{2,8},{3,9},{4,10},{5,11},{6,12},{7,13}}"}
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(14)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(13)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r), control-predecessors={constant.2559}
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(cp, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=2
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(cp, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k), control-predecessors={a}
}
ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  config_.set_num_partitions(8);
  config_.set_replica_count(4);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, /*level_to_operate_on=*/0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   /*direction=*/
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   /*should_process=*/IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
  // CHECK: %while_body
  // CHECK: %[[cp:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}_xla_send_recv_validation={{[{]}}{0,6},{1,7},{2,8},{3,9},{4,10},{5,11},{6,12},{7,12}{{[}]}}}
  // CHECK: %[[dus:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[cp]], {{.*}})
  // CHECK: ROOT {{.+}} = {{.+}} tuple({{.*}}%[[dus]], {{.*}})
  // CHECK: ENTRY %entry
  // CHECK: %[[while:.+]] = {{.+}} while({{.*}})
  // CHECK: %[[gte:.+]] = {{.+}} get-tuple-element({{.*}}%[[while]]), index=1
  // CHECK: %[[cp2:.+]] = {{.+}} collective-permute({{.*}}%[[gte]]), {{.+}}_xla_send_recv_validation={{[{]}}{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{0,0}{{[}]}}
  // CHECK: %[[dus:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[cp2]], {{.*}})
  // CHECK: %[[tuple:.+]] = {{.+}} tuple({{.*}}%[[dus]], {{.*}})
  // CHECK: ROOT {{.+}} = {{.+}} get-tuple-element({{.*}}%[[tuple]]), index=1
  )"));
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsCollectivePermuteBackwardCycle) {
  constexpr absl::string_view hlo_string = R"(
HloModule module
add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}
while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(14)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}
while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  cp = bf16[3,8,128] collective-permute(get-tuple-element.395), channel_id=1, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}},
                     frontend_attributes={_xla_send_recv_validation="{{7,13},{6,12},{5,11},{4,10},{3,9},{2,8},{1,7},{0,6}}"}
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(14)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(13)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r), control-predecessors={constant.2559}
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(cp, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=2
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(cp, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k), control-predecessors={a}
}
ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  config_.set_num_partitions(8);
  config_.set_replica_count(4);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, /*level_to_operate_on=*/0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   /*direction=*/
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   /*should_process=*/IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
  // CHECK: %while_body
  // CHECK: %[[cp:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}_xla_send_recv_validation={{[{]}}{7,12},{6,12},{5,11},{4,10},{3,9},{2,8},{1,7},{0,6}{{[}]}}}
  // CHECK: %[[dus:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[cp]], {{.*}})
  // CHECK: ROOT {{.+}} = {{.+}} tuple({{.*}}%[[dus]], {{.*}})
  // CHECK: ENTRY %entry
  // CHECK: %[[while:.+]] = {{.+}} while({{.+}})
  // CHECK: %[[gte:.+]] = {{.+}} get-tuple-element({{.*}}%[[while]]), index=1
  // CHECK: %[[cp2:.+]] = {{.+}} collective-permute({{.*}}%[[gte]]), {{.+}}_xla_send_recv_validation={{[{]}}{0,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0}{{[}]}}
  // CHECK: %[[dus:.+]] = {{.+}} dynamic-update-slice({{.*}}%[[cp2]], {{.*}})
  // CHECK: %[[tuple:.+]] = {{.+}} tuple({{.*}}%[[dus]], {{.*}})
  // CHECK: ROOT {{.+}} = {{.+}} get-tuple-element({{.*}}%[[tuple]]), index=1
  )"));
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsModifyOut) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  constant.10 = bf16[] constant(0)
  b = bf16[3,1,2,128] broadcast(constant.10), dimensions={}
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, b)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsPlusForward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) tuple(c0, p0, p1, p0)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/false, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsPlusForwardConvertOutput) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = f32[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  get-tuple-element.5 = f32[3,8,128] get-tuple-element(param), index=3
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = f32[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  cvt.0 = bf16[1,8,128] convert(dynamic-slice.99)
  mul = bf16[1,8,128] multiply(cvt.0, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  cvt.1 = f32[1,8,128] convert(ar.1)
  dynamic-update-slice.35 = f32[3,8,128] dynamic-update-slice(get-tuple-element.395, cvt.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = f32[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) tuple(c0, p0, p1, p0)
  while = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = f32[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/false, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwise) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwiseSortFormattingOps) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  negate.1 = bf16[1,8,128] negate(ar.1)
  negate.2 = bf16[1,8,128] negate(ar.1)
  add = bf16[1,8,128] multiply(negate.1, negate.2)
  mul3 = bf16[1,8,128] multiply(add, add)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, ElementWiseUser) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul2, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneNotFirstIdxSink) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  t2 = bf16[1,128,8] transpose(mul), dimensions={0,2,1}
  ar.1 = bf16[1,128,8] all-reduce(t2), replica_groups={}, to_apply=add, channel_id=1
  %c = bf16[] custom-call(), custom_call_target="Boh"
  %b = bf16[1,128,8] broadcast(c), dimensions={}
  %a = bf16[1,128,8] add(ar.1, b)
  %t = bf16[1,8,128] transpose(a), dimensions={0,2,1}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, t, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35), control-predecessors={select.1348}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  const HloComputation* comp = while_instr->while_body();
  const HloInstruction* root_loop = comp->root_instruction();
  EXPECT_TRUE(root_loop->HasControlDependencies());
  EXPECT_EQ(root_loop->control_predecessors().size(), 1);
  const HloInstruction* select_instr_loop =
      root_loop->control_predecessors()[0];
  const HloInstruction* transpose_instr_loop =
      FindInstruction(module.get(), HloOpcode::kTranspose);
  EXPECT_EQ(transpose_instr_loop->dimensions(),
            std::vector<int64_t>({0, 1, 3, 2}));
  EXPECT_EQ(select_instr_loop->opcode(), HloOpcode::kSelect);
}

TEST_F(CollectivePipelinerTest, ForwardSinkLinearShape4097) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,4097], bf16[3,4097]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,4097], bf16[3,4097]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,4097] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,4097] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,4097] dynamic-slice(get-tuple-element.35, select.1348, constant.2561), dynamic_slice_sizes={1,4097}
  mul = bf16[1,4097] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,4097] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  c = bf16[] custom-call(), custom_call_target="Boh"
  b = bf16[1,4097] broadcast(c), dimensions={}
  a = bf16[1,4097] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,4097] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561)
  ROOT tuple = (s32[], bf16[3,4097], bf16[3,4097]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35), control-predecessors={select.1348}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,4097] parameter(0)
  tuple = (s32[], bf16[3,4097], bf16[3,4097]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,4097], bf16[3,4097]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,4097] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/false,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  const HloComputation* comp = while_instr->while_body();
  const HloInstruction* root_loop = comp->root_instruction();
  EXPECT_TRUE(root_loop->HasControlDependencies());
  EXPECT_EQ(root_loop->control_predecessors().size(), 1);
  const HloInstruction* select_instr_loop =
      root_loop->control_predecessors()[0];
  EXPECT_EQ(select_instr_loop->opcode(), HloOpcode::kSelect);
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneNotFirstIdxSinkCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  %c = bf16[] custom-call(), custom_call_target="Boh"
  %b = bf16[1,8,128] broadcast(c), dimensions={}
  %a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/false,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* all_reduce = module->entry_computation()
                                         ->root_instruction()
                                         ->operand(0)
                                         ->operand(1)
                                         ->operand(0)
                                         ->operand(0);
  EXPECT_EQ(all_reduce->opcode(), HloOpcode::kAllReduce);
  EXPECT_EQ(all_reduce->shape().dimensions(0), 3);
}

// Checks that we shouldn't pipeline Send/Recv by accident while pipelining
// other collective, such as all-gather. In the test, the chain leading to
// all-gather contains Recv/Recv-done, which prevents us from pipelining the
// all-gather backward.
TEST_F(CollectivePipelinerTest, NotTransformAllGatherWithRecvInChainBackwards) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  after-all = token[] after-all()
  recv = (bf16[1,1,2,128], u32[], token[]) recv(after-all), channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}"
    }
  send = (bf16[1,1,2,128], u32[], token[]) send(get-tuple-element.k, after-all), channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}"
    }
  send-done = token[] send-done(send), channel_id=2
  recv-done = (bf16[1,1,2,128], token[]) recv-done(recv), channel_id=2
  recv-data = bf16[1,1,2,128] get-tuple-element(recv-done), index=0

  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(recv-data, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
}

TEST_F(CollectivePipelinerTest, TransformRecvSendBackwards) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(25)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    p = get-tuple-element(%param), index=1
    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}",
      _xla_send_recv_pipeline="0"
    }
    send = (f32[1, 1024, 1024], u32[], token[]) send(p, after-all), channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}",
      _xla_send_recv_pipeline="0"
    }
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=1, frontend_attributes={
       _xla_send_recv_pipeline="0"
    }
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0

    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}

    send-done = token[] send-done(send), channel_id=1, frontend_attributes={
       _xla_send_recv_pipeline="0"
    }
    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}
    while_init = (u32[], f32[1, 1024, 1024]) tuple(c0, init)
    while_result = (u32[], f32[1, 1024, 1024]) while(while_init), body=body, condition=cond, backend_config="{\"known_trip_count\":{\"n\":\"25\"}}"
    ROOT result = f32[1, 1024, 1024] get-tuple-element(while_result), index=1
  }
  )";

  auto should_pipeline = [](const HloInstruction* instruction) {
    if (!HloPredicateIsOp<HloOpcode::kRecvDone>(instruction)) return false;
    const HloRecvDoneInstruction* recv_done =
        dynamic_cast<const HloRecvDoneInstruction*>(instruction);
    if (recv_done->is_host_transfer()) return false;
    // Check that the recv-done is used for non-trivial computation, which can
    // also help avoid repeatedly pipelining a loop.
    return (recv_done->user_count() == 1 && recv_done->parent() != nullptr &&
            recv_done->users()[0] != recv_done->parent()->root_instruction());
  };
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   should_pipeline)
          .value());
  XLA_VLOG_LINES(10, module->ToString());
  auto recv1 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.1"));
  EXPECT_NE(recv1, nullptr);
  auto recv2 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.2"));
  EXPECT_NE(recv2, nullptr);
  EXPECT_EQ(recv1->channel_id(), recv2->channel_id());

  auto send1 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.1"));
  EXPECT_NE(send1, nullptr);
  auto send2 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.2"));
  EXPECT_NE(send2, nullptr);
  EXPECT_EQ(send1->channel_id(), send2->channel_id());

  EXPECT_EQ(recv1->channel_id(), send1->channel_id());
}

TEST_F(CollectivePipelinerTest,
       TransformRecvSendBackwardsWithLoopVariantParameter) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    after-all.0 = token[] after-all()
    recv.0 = (u32[2], u32[], token[]) recv(after-all.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_other_attr="0"
      }
    after-all.0.s = token[] after-all()
    send.0 = (u32[2], u32[], token[]) send(send-data, after-all.0.s),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_other_attr="0"
      }
    recv-done.0 = (u32[2], token[]) recv-done(recv.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    recv-data = u32[2] get-tuple-element(recv-done.0), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    send-done.0 = token[] send-done(send.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  auto should_pipeline = [](const HloInstruction* instr) {
    if (!HloPredicateIsOp<HloOpcode::kRecv>(instr) &&
        !HloPredicateIsOp<HloOpcode::kSend>(instr))
      return false;
    const HloSendRecvInstruction* send_recv =
        dynamic_cast<const HloSendRecvInstruction*>(instr);
    // Check that the Send or Recv is used for non-trivial computation, which
    // also help avoid repeatedly pipelining a loop.
    return (send_recv->user_count() == 1 && send_recv->parent() != nullptr &&
            send_recv->users()[0] != send_recv->parent()->root_instruction());
  };
  auto should_allow_loop_variant_parameter = [](const HloInstruction* instr) {
    CHECK(instr->opcode() == HloOpcode::kGetTupleElement &&
          instr->operand(0)->opcode() == HloOpcode::kParameter);
    return true;
  };
  const char* kAttr = "_xla_other_attr";
  // Mutate an existing attribute.
  auto postprocess_peeled = [&](HloInstruction* instr,
                                HloInstruction* new_while_instr) {
    xla::FrontendAttributes attributes = instr->frontend_attributes();
    (*attributes.mutable_map())[kAttr] = "1";
    instr->set_frontend_attributes(attributes);
    return absl::OkStatus();
  };
  auto postprocess_rotated = [&](HloInstruction* instr,
                                 HloInstruction* new_while_instr) {
    xla::FrontendAttributes attributes = instr->frontend_attributes();
    (*attributes.mutable_map())[kAttr] = "2";
    instr->set_frontend_attributes(attributes);
    return absl::OkStatus();
  };
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   should_pipeline,
                   /*acceptable_formatting=*/HloPredicateTrue,
                   /*reuse_pipelined_op_buffer=*/HloPredicateTrue,
                   should_allow_loop_variant_parameter, postprocess_peeled,
                   postprocess_rotated)
          .value());
  XLA_VLOG_LINES(10, module->ToString());
  auto while_op = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_op->opcode(), HloOpcode::kWhile);
  EXPECT_EQ(while_op->shape().tuple_shapes().size(), 5);
  auto recv1 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.1"));
  EXPECT_NE(recv1, nullptr);
  auto recv2 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.2"));
  EXPECT_NE(recv2, nullptr);
  EXPECT_EQ(recv1->channel_id(), recv2->channel_id());

  auto send1 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.1"));
  EXPECT_NE(send1, nullptr);
  auto send2 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.2"));
  EXPECT_NE(send2, nullptr);
  EXPECT_EQ(send1->channel_id(), send2->channel_id());

  EXPECT_EQ(recv1->channel_id(), send1->channel_id());

  const char* kSourceTarget = "_xla_send_recv_source_target_pairs={{3,0}}";
  const char* kPeeledAttr = "_xla_other_attr=\"1\"";
  const char* kRotatedAttr = "_xla_other_attr=\"2\"";
  EXPECT_THAT(send1->ToString(), ::testing::HasSubstr(kSourceTarget));
  EXPECT_THAT(recv1->ToString(), ::testing::HasSubstr(kSourceTarget));
  EXPECT_THAT(send2->ToString(), ::testing::HasSubstr(kSourceTarget));
  EXPECT_THAT(recv2->ToString(), ::testing::HasSubstr(kSourceTarget));
  EXPECT_THAT(send1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(recv1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(send2->ToString(), ::testing::HasSubstr(kRotatedAttr));
  EXPECT_THAT(recv2->ToString(), ::testing::HasSubstr(kRotatedAttr));
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwiseMerge) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ar.2 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.2)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwiseFeedTwo) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ar.2 = bf16[1,8,128] all-reduce(ar.1), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, bc), control-predecessors={ar.1}
  mul3 = bf16[1,8,128] multiply(mul2, ar.2)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5), control-predecessors={ar.1}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwiseFeedTwoWithReduce) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.2 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  bm = bf16[1,1,8,128] broadcast(mul), dimensions={1,2,3}
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,1,8,128] all-reduce(bm), replica_groups={}, to_apply=add, channel_id=1
  ar.2 = bf16[1,1,8,128] all-reduce(ar.1), replica_groups={}, to_apply=add, channel_id=2
  red.1 = bf16[1,8,128] reduce(ar.1, c2), to_apply=add.1, dimensions={0}
  red.2 = bf16[1,8,128] reduce(ar.2, c2), to_apply=add.2, dimensions={0}
  mul2 = bf16[1,8,128] multiply(red.1, bc), control-predecessors={ar.1}
  mul3 = bf16[1,8,128] multiply(mul2, red.2)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5), control-predecessors={ar.1}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, PipelinedReduceScatterCanPassVerifier) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

to_apply0 {
  Arg_0.732 = bf16[] parameter(0)
  Arg_1.733 = bf16[] parameter(1)
  ROOT add.734 = bf16[] add(Arg_0.732, Arg_1.733)
}

body {
  p2 = (s32[], bf16[3,4096,4096]{2,1,0}, bf16[10,512,3,4096]{3,2,1,0}) parameter(0)
  gte2 = bf16[3,4096,4096]{2,1,0} get-tuple-element(p2), index=1
  gte3 = bf16[10,512,3,4096]{3,2,1,0} get-tuple-element(p2), index=2
  c2 = s32[] constant(9)
  gte4 = s32[] get-tuple-element(p2), index=0
  sub0 = s32[] subtract(c2, gte4)
  c3 = s32[] constant(0)
  comp1 = pred[] compare(sub0, c3), direction=LT
  c4 = s32[] constant(19)
  sub2 = s32[] subtract(c4, gte4)
  sel0 = s32[] select(comp1, sub2, sub0)

  rsp0 = bf16[3,4096,4096]{2,1,0} reshape(gte2)
  rs0 = bf16[3,4096,512]{2,1,0} reduce-scatter(rsp0), channel_id=75, replica_groups={{0,1,2,3}}, dimensions={2}, to_apply=to_apply0
  tran0 = bf16[512,3,4096]{0,2,1} transpose(rs0), dimensions={2,0,1}
  rsp1 = bf16[1,512,3,4096]{3,2,1,0} reshape(tran0)
  dus0 = bf16[10,512,3,4096]{3,2,1,0} dynamic-update-slice(gte3, rsp1, sel0, c3, c3, /*index=5*/c3)
  c5 = s32[] constant(1)
  add0 = s32[] add(gte4, c5)
  ROOT t1 = (s32[], bf16[3,4096,4096]{2,1,0}, bf16[10,512,3,4096]{3,2,1,0}) tuple(add0, rsp0, dus0)
} // body

condition {
  cond_p1 = (s32[], bf16[3,4096,4096]{2,1,0}, bf16[10,512,3,4096]{3,2,1,0}) parameter(0)
  gte1 = s32[] get-tuple-element(cond_p1), index=0
  c1 = s32[] constant(9)
  ROOT comp0 = pred[] compare(gte1, c1), direction=LT
}

ENTRY main.3813_spmd {
  p0 = bf16[3,4096,4096]{2,1,0} parameter(0)
  p1 = bf16[10,512,3,4096]{3,2,1,0} parameter(1)
  c0 = s32[] constant(0)

  t0 = (s32[], bf16[3,4096,4096]{2,1,0}, bf16[10,512,3,4096]{3,2,1,0}) tuple(c0, p0, p1)
  w0 = (s32[], bf16[3,4096,4096]{2,1,0}, bf16[10,512,3,4096]{3,2,1,0}) while(t0), condition=condition, body=body
  ROOT gte0 = bf16[3,4096,4096]{2,1,0} get-tuple-element(w0), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/true,
                   /*process_different_sized_ops=*/true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   HloPredicateIsOp<HloOpcode::kReduceScatter>)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision*/ true);
  ASSERT_IS_OK(verifier.Run(module.get()).status());
}

TEST_F(CollectivePipelinerTest,
       PipelineBackwardIncludeInvariantMultiConsumerInChain) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

while_cond {
  param = (s32[], bf16[1,8,2048,32768]{3,2,1,0}, bf16[1,8,2048,32768]{3,2,1,0}) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[1,8,2048,32768]{3,2,1,0}, bf16[1,8,2048,32768]{3,2,1,0}) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[1,8,2048,32768]{3,2,1,0} get-tuple-element(param), index=1
  get-tuple-element.397 = bf16[1,8,2048,32768]{3,2,1,0} get-tuple-element(param), index=2

  constant.1 = bf16[] constant(2)
  broadcast.3593 = bf16[1,8,2048,32768]{3,2,1,0} broadcast(constant.1), dimensions={}

  add.2 = bf16[1,8,2048,32768]{3,2,1,0} add(broadcast.3593, get-tuple-element.395)

  all-gather.1 = bf16[1,64,2048,32768]{3,2,1,0} all-gather(broadcast.3593), channel_id=1, dimensions={1}, replica_groups={}

  slice.2 = bf16[1,8,2048,32768]{3,2,1,0} slice(all-gather.1), slice={[0:1], [8:16], [0:2048], [0:32768]}
  constant.2 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2)

  ROOT tuple = (s32[], bf16[1,8,2048,32768]{3,2,1,0}, bf16[1,8,2048,32768]{3,2,1,0}) tuple(add.230, add.2, slice.2)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[1,8,2048,32768]{3,2,1,0} parameter(0)
  p1 = bf16[1,8,2048,32768]{3,2,1,0} parameter(1)

  tuple = (s32[], bf16[1,8,2048,32768]{3,2,1,0}, bf16[1,8,2048,32768]{3,2,1,0}) tuple(c0, p0, p1)
  while = (s32[], bf16[1,8,2048,32768]{3,2,1,0}, bf16[1,8,2048,32768]{3,2,1,0}) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[1,8,2048,32768]{3,2,1,0} get-tuple-element(while), index=1
}
)";

  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();

  EXPECT_TRUE(
      RunOptimizer(
          module.get(), /*last_run=*/true, 0,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/false,
          /*direction=*/
          collective_pipeliner_utils::PipeliningDirection::kBackward,
          /*should_process=*/IsAllGather,
          /*acceptable_formatting=*/HloPredicateTrue,
          /*reuse_pipelined_op_buffer=*/HloPredicateTrue,
          /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateTrue,
          /*postprocess_backward_peeled=*/{},
          /*postprocess_backward_rotated=*/{},
          /*postprocess_backward_peeled_trailing=*/{},
          /*should_add_loop_invariant_op_in_chain=*/true)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  // Expect the while instruction input tuple to have 5 operands instead of 3
  // operands. 4th operand should be the peeled allgather in the main
  // computation.
  EXPECT_THAT(while_instr, op::While(op::Tuple(_, _, _, op::AllGather(), _)));

  HloInstruction* root = while_instr->while_body()->root_instruction();
  // Expect the while loop now to have 5 operands at the root instead of 3
  // operands. 4th operands should be the pipelined allgather.
  EXPECT_THAT(root, op::Tuple(_, _, _, op::AllGather(), _));

  // Now we turn should_add_loop_invariant_op_in_chain off, the hlo shouldn't
  // change at all.
  auto ref_module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(
          ref_module.get(), /*last_run=*/true, 0,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/false,
          /*direction=*/
          collective_pipeliner_utils::PipeliningDirection::kBackward,
          /*should_process=*/IsAllGather,
          /*acceptable_formatting=*/HloPredicateTrue,
          /*reuse_pipelined_op_buffer=*/HloPredicateTrue,
          /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateTrue,
          /*postprocess_backward_peeled=*/{},
          /*postprocess_backward_rotated=*/{},
          /*postprocess_backward_peeled_trailing=*/{},
          /*should_add_loop_invariant_op_in_chain=*/false)
          .value());
}

TEST_F(CollectivePipelinerTest, BroadcastAsFormattingOp) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  slice = bf16[1,8,120] slice(ar.1), slice={[0:1], [0:8], [0:120]}
  constant.2563 = bf16[] constant(5.0)
  pad = bf16[1,8,128] pad(slice, constant.2563), padding=0_0x0_0x0_8
  b.1 = bf16[1,8,128,32] broadcast(pad), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  EXPECT_GE(while_instr->users().size(), 2);
  EXPECT_TRUE(
      absl::c_any_of(while_instr->users(), [](const HloInstruction* user) {
        return absl::c_any_of(
            user->users(), [](const HloInstruction* user_user) {
              return user_user->opcode() == HloOpcode::kAllReduce;
            });
      }));
}

TEST_F(CollectivePipelinerTest, ForwardSinkDependentPipelineableCollectives) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  ar.2 = bf16[1,8,128] all-reduce(reduce), replica_groups={}, to_apply=add, channel_id=2
  c1 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c1)
  mul1 = bf16[1,8,128] multiply(ar.2, bc)
  mul3 = bf16[1,8,128] multiply(mul1, ar.2)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul3, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  config_.set_use_spmd_partitioning(true);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(
          module.get(), /*last_run=*/true,
          /*level_to_operate_on=*/0,
          /*pipeline_use_tree=*/true,
          /*process_different_sized_ops=*/true,
          collective_pipeliner_utils::PipeliningDirection::kForwardSink,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
          /*acceptable_formatting=*/HloPredicateIsNotOp<HloOpcode::kAllReduce>)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  // Return the closest all-reduce in the user subtree rooted at instruction i.
  std::function<const HloInstruction*(const HloInstruction*)> find_all_reduce =
      [&](const HloInstruction* i) -> const HloInstruction* {
    std::queue<const HloInstruction*> queue;
    queue.push(i);
    absl::flat_hash_set<HloInstruction*> visited;
    while (!queue.empty()) {
      const HloInstruction* curr_inst = queue.front();
      queue.pop();
      for (HloInstruction* operand : curr_inst->operands()) {
        if (operand->opcode() == HloOpcode::kAllReduce) {
          return operand;
        }
        if (visited.insert(operand).second) {
          queue.push(operand);
        }
      }
    }
    return nullptr;
  };
  // Check if root has the two all-reduces in the operand subtree where one is
  // an ancestor of the other.
  const HloInstruction* all_reduce1 =
      find_all_reduce(module->entry_computation()->root_instruction());
  EXPECT_NE(all_reduce1, nullptr);
  const HloInstruction* all_reduce2 = find_all_reduce(all_reduce1);
  EXPECT_NE(all_reduce2, nullptr);
  EXPECT_THAT(all_reduce2, op::AllReduce(op::GetTupleElement(op::While())));
}

TEST_F(CollectivePipelinerTest,
       ForwardSinkDependentPipelineableCollectivesNotLastRun) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  ar.2 = bf16[1,8,128] all-reduce(reduce), replica_groups={}, to_apply=add, channel_id=2
  c1 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c1)
  mul1 = bf16[1,8,128] multiply(ar.2, bc)
  mul3 = bf16[1,8,128] multiply(mul1, ar.2)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul3, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  config_.set_use_spmd_partitioning(true);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(
          module.get(), /*last_run=*/false,
          /*level_to_operate_on=*/0,
          /*pipeline_use_tree=*/true,
          /*process_different_sized_ops=*/true,
          collective_pipeliner_utils::PipeliningDirection::kForwardSink,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
          /*acceptable_formatting=*/HloPredicateIsNotOp<HloOpcode::kAllReduce>)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  // Return the closest all-reduce in the user subtree rooted at instruction i.
  std::function<const HloInstruction*(const HloInstruction*)> find_all_reduce =
      [&](const HloInstruction* i) -> const HloInstruction* {
    std::queue<const HloInstruction*> queue;
    queue.push(i);
    absl::flat_hash_set<HloInstruction*> visited;
    while (!queue.empty()) {
      const HloInstruction* curr_inst = queue.front();
      queue.pop();
      for (HloInstruction* operand : curr_inst->operands()) {
        if (operand->opcode() == HloOpcode::kAllReduce) {
          return operand;
        }
        if (visited.insert(operand).second) {
          queue.push(operand);
        }
      }
    }
    return nullptr;
  };
  // Check if root has the two all-reduces in the operand subtree where one is
  // an ancestor of the other.
  const HloInstruction* all_reduce1 =
      find_all_reduce(module->entry_computation()->root_instruction());
  EXPECT_NE(all_reduce1, nullptr);
  const HloInstruction* all_reduce2 = find_all_reduce(all_reduce1);
  EXPECT_NE(all_reduce2, nullptr);
  EXPECT_THAT(all_reduce2, op::AllReduce(op::GetTupleElement(op::While())));
  // The root of while body should have a dynamic-update-slice operand which has
  // a custom call at operand index 1.
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  CHECK_NE(while_instr, nullptr);
  const HloInstruction* dynamic_update_slice =
      while_instr->while_body()->root_instruction()->operands().back();
  CHECK_EQ(dynamic_update_slice->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* custom_call = dynamic_update_slice->operand(1);
  CHECK(custom_call->IsCustomCall("SunkByPreviousStep"));
}

TEST_F(CollectivePipelinerTest,
       ForwardSinkDependentPipelineableCollectivesDoNotPipeline) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs.1 = bf16[] parameter(0)
  rhs.1 = bf16[] parameter(1)
  ROOT add.1 = bf16[] add(lhs.1, rhs.1)
}

while_body.clone {
  sink_param.1 = (s32[], bf16[3,8,128]{2,1,0}, bf16[3,8,128]{2,1,0}, bf16[3,1,8,128]{3,2,1,0}) parameter(0)
  get-tuple-element.0 = s32[] get-tuple-element(sink_param.1), index=0
  constant.5 = s32[] constant(1)
  add.2 = s32[] add(get-tuple-element.0, constant.5)
  get-tuple-element.1 = bf16[3,8,128]{2,1,0} get-tuple-element(sink_param.1), index=1
  get-tuple-element.2 = bf16[3,8,128]{2,1,0} get-tuple-element(sink_param.1), index=2
  get-tuple-element.3 = bf16[3,1,8,128]{3,2,1,0} get-tuple-element(sink_param.1), index=3
  constant.6 = s32[] constant(3)
  subtract.0 = s32[] subtract(constant.6, get-tuple-element.0)
  constant.7 = s32[] constant(-1)
  add.3 = s32[] add(subtract.0, constant.7)
  constant.8 = s32[] constant(0)
  compare.0 = pred[] compare(add.3, constant.8), direction=LT
  constant.9 = s32[] constant(2)
  add.4 = s32[] add(subtract.0, constant.9)
  select.0 = s32[] select(compare.0, add.4, add.3)
  dynamic-slice.0 = bf16[1,8,128]{2,1,0} dynamic-slice(get-tuple-element.2, select.0, constant.8, constant.8), dynamic_slice_sizes={1,8,128}
  mul.1 = bf16[1,8,128]{2,1,0} multiply(dynamic-slice.0, dynamic-slice.0)
  ar.0 = bf16[1,8,128]{2,1,0} all-reduce(mul.1), channel_id=1, replica_groups={}, to_apply=add
  b.0 = bf16[1,8,128,32]{3,2,1,0} broadcast(ar.0), dimensions={0,1,2}
  constant.10 = bf16[] constant(0)
  reduce.1 = bf16[1,8,128]{2,1,0} reduce(b.0, constant.10), dimensions={3}, to_apply=add.1
  reshape.1 = bf16[1,1,8,128]{3,2,1,0} reshape(reduce.1)
  custom-call.2 = bf16[1,1,8,128]{3,2,1,0} custom-call(reshape.1), custom_call_target="SunkByPreviousStep"
  constant.12 = s32[] constant(0)
  dynamic-update-slice.1 = bf16[3,1,8,128]{3,2,1,0} dynamic-update-slice(get-tuple-element.3, custom-call.2, select.0, constant.12, constant.12, constant.12)
  ROOT tuple.3 = (s32[], bf16[3,8,128]{2,1,0}, bf16[3,8,128]{2,1,0}, bf16[3,1,8,128]{3,2,1,0}) tuple(add.2, get-tuple-element.1, get-tuple-element.2, dynamic-update-slice.1)
}

while_cond.clone {
  sink_param = (s32[], bf16[3,8,128]{2,1,0}, bf16[3,8,128]{2,1,0}, bf16[3,1,8,128]{3,2,1,0}) parameter(0)
  gte.1 = s32[] get-tuple-element(sink_param), index=0
  constant.13 = s32[] constant(3)
  ROOT cmp.1 = pred[] compare(gte.1, constant.13), direction=LT
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128]{2,1,0} parameter(0)
  constant.2 = bf16[] constant(0)
  broadcast = bf16[3,1,8,128]{3,2,1,0} broadcast(constant.2), dimensions={}
  tuple.2 = (s32[], bf16[3,8,128]{2,1,0}, bf16[3,8,128]{2,1,0}, bf16[3,1,8,128]{3,2,1,0}) tuple(c0, p0, p0, broadcast)
  while.1 = (s32[], bf16[3,8,128]{2,1,0}, bf16[3,8,128]{2,1,0}, bf16[3,1,8,128]{3,2,1,0}) while(tuple.2), condition=while_cond.clone, body=while_body.clone
  get-tuple-element.5 = s32[] get-tuple-element(while.1), index=0
  get-tuple-element.4 = bf16[3,1,8,128]{3,2,1,0} get-tuple-element(while.1), index=3
  ar.4 = bf16[3,1,8,128]{3,2,1,0} all-reduce(get-tuple-element.4), channel_id=3, replica_groups={}, to_apply=add
  c1.3 = bf16[] constant(2)
  broadcast.1 = bf16[3,1,8,128]{3,2,1,0} broadcast(c1.3), dimensions={}
  mul1.2 = bf16[3,1,8,128]{3,2,1,0} multiply(ar.4, broadcast.1)
  mul3.2 = bf16[3,1,8,128]{3,2,1,0} multiply(mul1.2, ar.4)
  reshape.2 = bf16[3,8,128]{2,1,0} reshape(mul3.2)
  get-tuple-element.6 = bf16[3,8,128]{2,1,0} get-tuple-element(while.1), index=2
  tuple.4 = (s32[], bf16[3,8,128]{2,1,0}, bf16[3,8,128]{2,1,0}) tuple(get-tuple-element.5, reshape.2, get-tuple-element.6)
  ROOT gte1 = bf16[3,8,128]{2,1,0} get-tuple-element(tuple.4), index=1
}
)";
  config_.set_use_spmd_partitioning(true);
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(
          module.get(), /*last_run=*/false,
          /*level_to_operate_on=*/0,
          /*pipeline_use_tree=*/true,
          /*process_different_sized_ops=*/true,
          /*direction=*/
          collective_pipeliner_utils::PipeliningDirection::kForwardSink,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
          /*acceptable_formatting=*/HloPredicateIsNotOp<HloOpcode::kAllReduce>,
          /*reuse_pipelined_op_buffer=*/HloPredicateTrue,
          /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
          /*postprocess_backward_peeled=*/{},
          /*postprocess_backward_rotated=*/{},
          /*postprocess_backward_peeled_trailing=*/{},
          /*should_add_loop_invariant_op_in_chain=*/false,
          /*collective_size_threshold_to_stop_sinking=*/1024)
          .value());
}

TEST_F(CollectivePipelinerTest, ForwardSinkFirstDimNotMatchingLoopCount) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[5,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[5,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[5,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  c = bf16[] custom-call(), custom_call_target="Boh"
  b = bf16[1,8,128] broadcast(c), dimensions={}
  a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[5,8,128] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[5,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35), control-predecessors={select.1348}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[5,8,128] parameter(0)
  p1 = bf16[3,8,128] parameter(1)
  tuple = (s32[], bf16[5,8,128], bf16[3,8,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[5,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[5,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(
          module.get(), /*last_run=*/true,
          /*level_to_operate_on=*/0,
          /*pipeline_use_tree=*/true,
          /*process_different_sized_ops=*/true,
          collective_pipeliner_utils::PipeliningDirection::kForwardSink)
          .value());
}

TEST_F(CollectivePipelinerTest, ForwardSinkNotFirstDim) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  %c = bf16[] custom-call(), custom_call_target="Boh"
  %b = bf16[1,8,128] broadcast(c), dimensions={}
  %a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, a, constant.2561, select.1348, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35), control-predecessors={select.1348}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(
          module.get(), /*last_run=*/true,
          /*level_to_operate_on=*/0,
          /*pipeline_use_tree=*/true,
          /*process_different_sized_ops=*/true,
          collective_pipeliner_utils::PipeliningDirection::kForwardSink)
          .value());
}

TEST_F(CollectivePipelinerTest, CollectiveWithMultipleDUS) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  EXPECT_TRUE(
      absl::c_any_of(while_instr->users(), [](const HloInstruction* user) {
        return absl::c_any_of(
            user->users(), [](const HloInstruction* user_user) {
              return user_user->opcode() == HloOpcode::kAllReduce;
            });
      }));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kGetTupleElement);
  const HloInstruction* new_tuple =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(new_tuple->opcode(), HloOpcode::kTuple);
  // There should be two reshapes in this tuple (replacing the two
  // dynamic-update-slices).
  EXPECT_EQ(absl::c_count_if(new_tuple->operands(),
                             [](const HloInstruction* operand) {
                               return operand->opcode() == HloOpcode::kReshape;
                             }),
            2);
}

TEST_F(CollectivePipelinerTest, CollectiveWithMultipleDUSNotLastRun) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/false,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  CHECK_NE(while_instr, nullptr);
  EXPECT_TRUE(
      absl::c_any_of(while_instr->users(), [](const HloInstruction* user) {
        return absl::c_any_of(
            user->users(), [](const HloInstruction* user_user) {
              return user_user->opcode() == HloOpcode::kAllReduce;
            });
      }));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kGetTupleElement);
  const HloInstruction* new_tuple =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(new_tuple->opcode(), HloOpcode::kTuple);
  // There should be two reshapes in this tuple (replacing the two
  // dynamic-update-slices).
  EXPECT_EQ(absl::c_count_if(new_tuple->operands(),
                             [](const HloInstruction* operand) {
                               return operand->opcode() == HloOpcode::kReshape;
                             }),
            2);
  // The root of while body should have a dynamic-update-slice operand which has
  // a custom call at operand index 1.
  const HloInstruction* dynamic_update_slice =
      while_instr->while_body()->root_instruction()->operand(4);
  CHECK_EQ(dynamic_update_slice->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* custom_call = dynamic_update_slice->operand(1);
  CHECK(custom_call->IsCustomCall("SunkByPreviousStep"));
}

TEST_F(CollectivePipelinerTest, CollectiveWithMultipleDUSSameBuffer) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(
      RunOptimizer(
          module.get(), /*last_run=*/true,
          /*level_to_operate_on=*/0,
          /*pipeline_use_tree=*/true,
          /*process_different_sized_ops=*/true,
          collective_pipeliner_utils::PipeliningDirection::kForwardSink)
          .value());
}

TEST_F(CollectivePipelinerTest, MergeTwoCollectivesEachWithTwoDUS) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.397 = bf16[3,8,128] get-tuple-element(param), index=3
  get-tuple-element.398 = bf16[3,8,128] get-tuple-element(param), index=4
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=5
  get-tuple-element.36 = bf16[3,8,128] get-tuple-element(param), index=6
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  // ar.1 is used by dynamic-update-slice.35 and dynamic-update-slice.36
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, mul4, select.1348, constant.2561, constant.2561)

  // ar.1 is used by dynamic-update-slice.37 and dynamic-update-slice.38
  // dynamic-update-slice.37 actually uses both ar.1 and ar.2
  dynamic-slice.100 = bf16[1,8,128] dynamic-slice(get-tuple-element.36, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul.1 = bf16[1,8,128] multiply(dynamic-slice.100, dynamic-slice.99)
  ar.2 = bf16[1,8,128] all-reduce(mul.1), replica_groups={}, to_apply=add, channel_id=1
  divide = bf16[1,8,128] divide(ar.1, ar.2)
  dynamic-update-slice.37 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.397, divide, select.1348, constant.2561, constant.2561)
  mul.2 = bf16[1,8,128] multiply(ar.2, ar.2)
  abs = bf16[1,8,128] abs(mul.2)
  dynamic-update-slice.38 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.398, abs, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, dynamic-update-slice.37, dynamic-update-slice.38, get-tuple-element.35, get-tuple-element.36)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,8,128] parameter(1)
  p2 = bf16[3,8,128] parameter(2)
  p3 = bf16[3,8,128] parameter(3)
  p4 = bf16[3,8,128] parameter(4)
  p5 = bf16[3,8,128] parameter(5)

  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p1, p2, p3, p4, p5)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::Tuple(
                  op::GetTupleElement(op::While()), op::Reshape(op::Reduce()),
                  op::Reshape(op::Multiply()), op::Reshape(op::Divide()),
                  op::Reshape(op::Abs()), op::GetTupleElement(op::While()),
                  op::GetTupleElement(op::While()))));
}

TEST_F(CollectivePipelinerTest, MergeTwoCollectivesEachWithTwoDUSNotLastRun) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.397 = bf16[3,8,128] get-tuple-element(param), index=3
  get-tuple-element.398 = bf16[3,8,128] get-tuple-element(param), index=4
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=5
  get-tuple-element.36 = bf16[3,8,128] get-tuple-element(param), index=6
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  // ar.1 is used by dynamic-update-slice.35 and dynamic-update-slice.36
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  b.1 = bf16[1,8,128,32] broadcast(ar.1), dimensions={0,1,2}
  constant = bf16[] constant(0)
  reduce = bf16[1,8,128] reduce(b.1, constant), dimensions={3}, to_apply=add.1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, reduce, select.1348, constant.2561, constant.2561)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, mul4, select.1348, constant.2561, constant.2561)

  // ar.1 is used by dynamic-update-slice.37 and dynamic-update-slice.38
  // dynamic-update-slice.37 actually uses both ar.1 and ar.2
  dynamic-slice.100 = bf16[1,8,128] dynamic-slice(get-tuple-element.36, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul.1 = bf16[1,8,128] multiply(dynamic-slice.100, dynamic-slice.99)
  ar.2 = bf16[1,8,128] all-reduce(mul.1), replica_groups={}, to_apply=add, channel_id=1
  divide = bf16[1,8,128] divide(ar.1, ar.2)
  dynamic-update-slice.37 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.397, divide, select.1348, constant.2561, constant.2561)
  mul.2 = bf16[1,8,128] multiply(ar.2, ar.2)
  abs = bf16[1,8,128] abs(mul.2)
  dynamic-update-slice.38 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.398, abs, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, dynamic-update-slice.37, dynamic-update-slice.38, get-tuple-element.35, get-tuple-element.36)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,8,128] parameter(1)
  p2 = bf16[3,8,128] parameter(2)
  p3 = bf16[3,8,128] parameter(3)
  p4 = bf16[3,8,128] parameter(4)
  p5 = bf16[3,8,128] parameter(5)

  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p1, p2, p3, p4, p5)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/false,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::Tuple(
                  op::GetTupleElement(op::While()), op::Reshape(op::Reduce()),
                  op::Reshape(op::Multiply()), op::Reshape(op::Divide()),
                  op::Reshape(op::Abs()), op::GetTupleElement(op::While()),
                  op::GetTupleElement(op::While()))));
  // The root of while body should have two dynamic-update-slice operands each
  // of which has a custom call at operand index 1.
  std::function<bool(const HloInstruction*)> is_dus_with_custom_call =
      [&](const HloInstruction* inst) -> bool {
    if (inst->opcode() != HloOpcode::kDynamicUpdateSlice) {
      return false;
    }
    return inst->operand(1)->IsCustomCall("SunkByPreviousStep");
  };
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  CHECK_NE(while_instr, nullptr);
  CHECK(is_dus_with_custom_call(
      while_instr->while_body()->root_instruction()->operand(7)));
  CHECK(is_dus_with_custom_call(
      while_instr->while_body()->root_instruction()->operand(8)));
}

// There is only one group of collectives in the following graph. The algorithm
// - first creates two individual groups: one for ar.1 and one for ar.2,
// - and then merges the two groups into one group because of ar.3
// - and then adds ar.4 into the existing group.
//  ar.1      ar.3      ar.2      ar.4
//      \    /    \    /    \    /
//       add.1     add.2     add.3
//         |         |         |
//      dus.35    dus.36    dus.37
TEST_F(CollectivePipelinerTest, MergeFourCollectives) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=3
  get-tuple-element.397 = bf16[3,8,128] get-tuple-element(param), index=4

  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)

  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ar.2 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=2
  ar.3 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=3
  ar.4 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=4

  add.1 = bf16[1,8,128] add(ar.1, ar.3)
  add.2 = bf16[1,8,128] add(ar.2, ar.3)
  add.3 = bf16[1,8,128] add(ar.2, ar.4)

  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, add.1, select.1348, constant.2561, constant.2561)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, add.2, select.1348, constant.2561, constant.2561)
  dynamic-update-slice.37 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.397, add.3, select.1348, constant.2561, constant.2561)

  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, get-tuple-element, dynamic-update-slice.35, dynamic-update-slice.36, dynamic-update-slice.37), control-predecessors={select.1348}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             [](const HloInstruction* instr) {
                               return instr->opcode() == HloOpcode::kAllReduce;
                             }),
            4);
}

TEST_F(CollectivePipelinerTest, NoRedundantBroadcastsInFormattingOps) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

add.1 {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,8,128] get-tuple-element(param), index=2
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  convert = bf16[] convert(add.232)
  broadcast = bf16[1,8,128] broadcast(convert)
  add.1 = bf16[1,8,128] add(ar.1, broadcast)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, add.1, select.1348, constant.2561, constant.2561)
  ar.2 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add.1, channel_id=2
  add.2 = bf16[1,8,128] add(ar.2, broadcast)
  dynamic-update-slice.36 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.396, add.2, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-update-slice.36, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(
                  module.get(), /*last_run=*/true,
                  /*level_to_operate_on=*/0,
                  /*pipeline_use_tree=*/true,
                  /*process_different_sized_ops=*/true,
                  collective_pipeliner_utils::PipeliningDirection::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  // There should be only one broadcast instruction using a get-tuple-element
  // from the while instruction.
  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             [](const HloInstruction* instr) {
                               return instr->opcode() ==
                                          HloOpcode::kBroadcast &&
                                      instr->operand(0)->opcode() ==
                                          HloOpcode::kGetTupleElement &&
                                      instr->operand(0)->operand(0)->opcode() ==
                                          HloOpcode::kWhile;
                             }),
            1);
}

TEST_F(CollectivePipelinerTest, AllGatherAsFormattingOp) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, dimensions={1}, to_apply=add, channel_id=2
  ar.1 = bf16[1,1,128] all-reduce(rs.1), replica_groups={}, to_apply=add, channel_id=1
  ag.1 = bf16[1,8,128] all-gather(ar.1), replica_groups={}, dimensions={1}, channel_id=3
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0, false, true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   HloPredicateIsOp<HloOpcode::kAllReduce>,
                   /*acceptable_formatting=*/HloPredicateTrue,
                   /*reuse_pipelined_op_buffer=*/HloPredicateTrue)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicUpdateSlice(op::GetTupleElement(op::While()),
                                     op::AllGather(op::AllReduce()),
                                     op::GetTupleElement(op::While()),
                                     op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, PipelinedSchedulingAnnotationsForward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99), frontend_attributes={_scheduling_group_id="1"}
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, dimensions={1}, to_apply=add, channel_id=2
  ar.1 = bf16[1,1,128] all-reduce(rs.1), replica_groups={}, to_apply=add, channel_id=1, frontend_attributes={_scheduling_group_id="2"}
  ag.1 = bf16[1,8,128] all-gather(ar.1), replica_groups={}, dimensions={1}, channel_id=3, frontend_attributes={_scheduling_group_id="3"}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0, false, true,
                   collective_pipeliner_utils::PipeliningDirection::kForward,
                   HloPredicateIsOp<HloOpcode::kAllReduce>,
                   /*acceptable_formatting=*/HloPredicateTrue,
                   /*reuse_pipelined_op_buffer=*/HloPredicateTrue)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kMultiply) {
      TF_ASSERT_OK_AND_ASSIGN(std::optional<int64_t> id,
                              GetSchedulingAnnotationGroupId(instr));
      EXPECT_EQ(id, 4);
    } else if (instr->opcode() == HloOpcode::kAllReduce) {
      TF_ASSERT_OK_AND_ASSIGN(std::optional<int64_t> id,
                              GetSchedulingAnnotationGroupId(instr));
      EXPECT_EQ(id, 5);
    } else if (instr->opcode() == HloOpcode::kAllGather) {
      TF_ASSERT_OK_AND_ASSIGN(std::optional<int64_t> id,
                              GetSchedulingAnnotationGroupId(instr));
      EXPECT_EQ(id, 6);
    }
  }
}

TEST_F(CollectivePipelinerTest, PipelinedSchedulingAnnotationsBackward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k), frontend_attributes={_scheduling_group_id="1"}
  a = bf16[1,2,128] add(r, r), control-predecessors={constant.2559}
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}, frontend_attributes={_scheduling_group_id="2"}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1, frontend_attributes={_scheduling_group_id="3"}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k), control-predecessors={a}
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(
      RunOptimizer(module.get(), /*last_run=*/true, 0,
                   /*pipeline_use_tree=*/false,
                   /*process_different_sized_ops=*/false,
                   collective_pipeliner_utils::PipeliningDirection::kBackward,
                   IsAllGather)
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kReshape) {
      TF_ASSERT_OK_AND_ASSIGN(std::optional<int64_t> id,
                              GetSchedulingAnnotationGroupId(instr));
      EXPECT_EQ(id, 4);
    } else if (instr->opcode() == HloOpcode::kAllGather) {
      TF_ASSERT_OK_AND_ASSIGN(std::optional<int64_t> id,
                              GetSchedulingAnnotationGroupId(instr));
      EXPECT_EQ(id, 5);
    } else if (instr->opcode() == HloOpcode::kAllReduce) {
      TF_ASSERT_OK_AND_ASSIGN(std::optional<int64_t> id,
                              GetSchedulingAnnotationGroupId(instr));
      EXPECT_EQ(id, 6);
    }
  }
}

TEST_F(CollectivePipelinerTest,
       TransformForwardWithInconsistentSchedulingAnnotations) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,8,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}, frontend_attributes={_scheduling_group_id="123:-1"}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto direction = collective_pipeliner_utils::PipeliningDirection::kForward;
  EXPECT_FALSE(
      RunOptimizer(
          module.get(), /*last_run=*/true, 0,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/true, direction,
          [direction, is_all_gather = IsAllGatherExplicitPipeliningAnnotation](
              const HloInstruction* instr) {
            return is_all_gather(instr, direction);
          })
          .value());
  HloInstruction* ag = FindInstruction(module.get(), "ag.1");
  std::optional<Annotation> annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_TRUE(annotation->iteration_id);
  EXPECT_EQ(annotation->iteration_id->iteration_id, -1);

  LegalizeSchedulingAnnotations::Config config;
  config.remove_loop_iteration_annotation_only = true;
  EXPECT_TRUE(LegalizeSchedulingAnnotations(config).Run(module.get()).value());
  annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_FALSE(annotation->iteration_id);
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformForwardWithConsistentSchedulingAnnotations) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,8,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}, frontend_attributes={_scheduling_group_id="123:1"}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
  }

  ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto direction = collective_pipeliner_utils::PipeliningDirection::kForward;
  EXPECT_TRUE(
      RunOptimizer(
          module.get(), /*last_run=*/true, 0,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/true, direction,
          [direction, is_all_gather = IsAllGatherExplicitPipeliningAnnotation](
              const HloInstruction* instr) {
            return is_all_gather(instr, direction);
          })
          .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _, op::AllGather(op::GetTupleElement(op::While())),
                        op::GetTupleElement(), op::Constant(), op::Constant()));

  HloInstruction* ag = FindInstruction(module.get(), "ag.2");
  std::optional<Annotation> annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_TRUE(annotation->iteration_id);
  EXPECT_EQ(annotation->iteration_id->iteration_id, 1);

  LegalizeSchedulingAnnotations::Config config;
  config.remove_loop_iteration_annotation_only = true;
  EXPECT_TRUE(LegalizeSchedulingAnnotations(config).Run(module.get()).value());
  annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_FALSE(annotation->iteration_id);
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformBackwardWithInconsistentSchedulingAnnotations) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

while_cond {
  param = (s32[], bf16[5,8,128], bf16[5,1,2,128]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c4 = s32[] constant(4)
  ROOT cmp = pred[] compare(loop_index, c4), direction=LT
}

while_body {
  param = (s32[], bf16[5,8,128], bf16[5,1,2,128]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,8,128] get-tuple-element(param), index=1
  slice_input = bf16[5,1,2,128] get-tuple-element(param), index=2
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  c3 = s32[] constant(3)
  three_minus_loop_index = s32[] subtract(c3, loop_index)
  dynamic_slice = bf16[1,1,2,128] dynamic-slice(slice_input, three_minus_loop_index, c0, c0, c0), dynamic_slice_sizes={1,1,2,128}
  dynamic_slice_reshape = bf16[1,2,128] reshape(dynamic_slice)
  add = bf16[1,2,128] add(dynamic_slice_reshape, dynamic_slice_reshape), control-predecessors={c3}
  all_gather = bf16[1,8,128] all-gather(add), dimensions={1}, replica_groups={}, frontend_attributes={_scheduling_group_id="123:1"}
  updated_partial_output = bf16[5,8,128] dynamic-update-slice(partial_output, all_gather, three_minus_loop_index, c0, c0)
  ROOT tuple = (s32[], bf16[5,8,128], bf16[5,1,2,128]) tuple(next_loop_index, updated_partial_output, slice_input), control-predecessors={add}
}

ENTRY entry {
  c1 = s32[] constant(1)
  p0 = bf16[5,8,128] parameter(0)
  p1 = bf16[5,1,2,128] parameter(1)
  tuple = (s32[], bf16[5,8,128], bf16[5,1,2,128]) tuple(c1, p0, p1)
  while = (s32[], bf16[5,8,128], bf16[5,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto direction = collective_pipeliner_utils::PipeliningDirection::kBackward;
  EXPECT_FALSE(
      RunOptimizer(
          module.get(), /*last_run=*/true, 0,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/false, direction,
          [direction, is_all_gather = IsAllGatherExplicitPipeliningAnnotation](
              const HloInstruction* instr) {
            return is_all_gather(instr, direction);
          })
          .value());

  HloInstruction* ag = FindInstruction(module.get(), "all_gather");
  std::optional<Annotation> annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_TRUE(annotation->iteration_id);
  EXPECT_EQ(annotation->iteration_id->iteration_id, 1);

  LegalizeSchedulingAnnotations::Config config;
  config.remove_loop_iteration_annotation_only = true;
  EXPECT_TRUE(LegalizeSchedulingAnnotations(config).Run(module.get()).value());
  annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_FALSE(annotation->iteration_id);
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformBackwardWithConsistentSchedulingAnnotations) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

while_cond {
  param = (s32[], bf16[5,8,128], bf16[5,1,2,128]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c4 = s32[] constant(4)
  ROOT cmp = pred[] compare(loop_index, c4), direction=LT
}

while_body {
  param = (s32[], bf16[5,8,128], bf16[5,1,2,128]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,8,128] get-tuple-element(param), index=1
  slice_input = bf16[5,1,2,128] get-tuple-element(param), index=2
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  c3 = s32[] constant(3)
  three_minus_loop_index = s32[] subtract(c3, loop_index)
  dynamic_slice = bf16[1,1,2,128] dynamic-slice(slice_input, three_minus_loop_index, c0, c0, c0), dynamic_slice_sizes={1,1,2,128}
  dynamic_slice_reshape = bf16[1,2,128] reshape(dynamic_slice)
  add = bf16[1,2,128] add(dynamic_slice_reshape, dynamic_slice_reshape), control-predecessors={c3}
  all_gather = bf16[1,8,128] all-gather(add), dimensions={1}, replica_groups={}, frontend_attributes={_scheduling_group_id="123:-1"}
  updated_partial_output = bf16[5,8,128] dynamic-update-slice(partial_output, all_gather, three_minus_loop_index, c0, c0)
  ROOT tuple = (s32[], bf16[5,8,128], bf16[5,1,2,128]) tuple(next_loop_index, updated_partial_output, slice_input), control-predecessors={add}
}

ENTRY entry {
  c1 = s32[] constant(1)
  p0 = bf16[5,8,128] parameter(0)
  p1 = bf16[5,1,2,128] parameter(1)
  tuple = (s32[], bf16[5,8,128], bf16[5,1,2,128]) tuple(c1, p0, p1)
  while = (s32[], bf16[5,8,128], bf16[5,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  auto direction = collective_pipeliner_utils::PipeliningDirection::kBackward;
  EXPECT_TRUE(
      RunOptimizer(
          module.get(), /*last_run=*/true, 0,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/false, direction,
          [direction, is_all_gather = IsAllGatherExplicitPipeliningAnnotation](
              const HloInstruction* instr) {
            return is_all_gather(instr, direction);
          })
          .value());
  const HloInstruction* while_instr =
      FindInstruction(module.get(), HloOpcode::kWhile);
  const HloComputation* comp = while_instr->while_body();
  const HloInstruction* root_loop = comp->root_instruction();

  const HloInstruction* shifted_loop_counter = root_loop->operand(4);
  EXPECT_EQ(shifted_loop_counter->opcode(), HloOpcode::kAdd);
  const HloInstruction* loop_increment = shifted_loop_counter->operand(1);
  EXPECT_EQ(loop_increment->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(loop_increment->literal().IsEqualAt({}, 1));

  HloInstruction* ag = FindInstruction(module.get(), "all_gather.2");
  std::optional<Annotation> annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_TRUE(annotation->iteration_id);
  EXPECT_EQ(annotation->iteration_id->iteration_id, -1);

  LegalizeSchedulingAnnotations::Config config;
  config.remove_loop_iteration_annotation_only = true;
  EXPECT_TRUE(LegalizeSchedulingAnnotations(config).Run(module.get()).value());
  annotation = GetSchedulingAnnotation(ag).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_FALSE(annotation->iteration_id);
  XLA_VLOG_LINES(1, module->ToString());
}

}  // namespace
}  // namespace xla
