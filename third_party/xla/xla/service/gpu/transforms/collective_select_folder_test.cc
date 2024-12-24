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

#include "xla/service/gpu/transforms/collective_select_folder.h"

#include <initializer_list>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::HasSubstr;

class CollectiveSelectFolderTest : public HloTestBase {
 public:
  absl::Status ExpectNoTranform(absl::string_view hlo_template) {
    return RunAndCheckHloRewrite(hlo_template, CollectiveSelectFolder(),
                                 /*expect_change=*/false)
        .status();
  }
};

void VerifyDirectDataFeedSPMD(HloModule* module,
                              absl::string_view expected_fwd_operand,
                              absl::string_view expected_bwd_operand) {
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kCollectivePermute);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kCollectivePermute);
  // First cp is sending backward per template.
  EXPECT_THAT(root->operand(1)->operand(0)->name(),
              HasSubstr(expected_bwd_operand))
      << root->operand(1)->name() << " is expected to operate on "
      << expected_bwd_operand;
  // Second cp is sending forward per template.
  EXPECT_THAT(root->operand(2)->operand(0)->name(),
              HasSubstr(expected_fwd_operand))
      << root->operand(2)->name() << " is expected to operate on "
      << expected_fwd_operand;
}

// HLO segment as would be generated in SPMD pipeline containing two collective
// permutes forming a cycle.
const char* kSPMD2cp = R"(
  HloModule test
  ENTRY circular_exchange {
    in_tpl = (f32[16], f32[16]) parameter(0)
    fwd_data = f32[16]{0} get-tuple-element(in_tpl), index=0
    bwd_data = f32[16]{0} get-tuple-element(in_tpl), index=1

    c_first_id = u32[] constant($first_id_constant)
    c_last_id = u32[] constant($last_id_constant)
    repl_id = u32[] partition-id()

    pred_first_id = pred[] compare(repl_id, c_first_id), direction=$direction
    is_first = pred[] broadcast(pred_first_id), dimensions={}

    pred_last_id = pred[] compare(repl_id, c_last_id), direction=$direction
    is_last = pred[] broadcast(pred_last_id), dimensions={}

    // This is the select that we want to optimize away.
    data_snd = f32[16] select(is_last, bwd_data, fwd_data)

    bwd_data_rcv = f32[16] collective-permute(data_snd), channel_id=1,
        source_target_pairs=$backward_pairs
    fwd_data_rcv = f32[16] collective-permute(data_snd), channel_id=2,
        source_target_pairs=$forward_pairs
    ROOT data_rcv = f32[16] select(is_first, bwd_data_rcv, fwd_data_rcv)
  }
)";

TEST_F(CollectiveSelectFolderTest, SimpleForwardCycle) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSPMD2cp, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$first_id_constant", "0"},
                             {"$last_id_constant", "3"},
                             {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                             {"$backward_pairs", "{{3,0}}"},
                             {"$direction", "EQ"}}));

  VerifyDirectDataFeedSPMD(module.get(), "fwd_data", "bwd_data");
}

TEST_F(CollectiveSelectFolderTest, SimpleBackwardCycle) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSPMD2cp, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$first_id_constant", "3"},
                             {"$last_id_constant", "0"},
                             {"$forward_pairs", "{{3,2},{2,1},{1,0}}"},
                             {"$backward_pairs", "{{0,3}}"},
                             {"$direction", "EQ"}}));
  VerifyDirectDataFeedSPMD(module.get(), "fwd_data", "bwd_data");
}

TEST_F(CollectiveSelectFolderTest, CompareNEForwardCycle) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSPMD2cp, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$first_id_constant", "0"},
                             {"$last_id_constant", "3"},
                             {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                             {"$backward_pairs", "{{3,0}}"},
                             {"$direction", "NE"}}));
  // Compared with SimpleForwardCycle above, this test flips the condition
  // and therefore the data being forwarded.
  VerifyDirectDataFeedSPMD(module.get(), "bwd_data", "fwd_data");
}

// Forceful case when select constant is not equal to the backward edge.
// In this case, backward collective-permute is expected to be linked
// to fwd_data while forward collective-permute is expected remain linked
// to the select.
TEST_F(CollectiveSelectFolderTest, LastDeviceIdMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSPMD2cp, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$first_id_constant", "0"},
                             {"$last_id_constant", "2"},  // mismatch
                             {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                             {"$backward_pairs", "{{3,0}}"},
                             {"$direction", "EQ"}}));
  VerifyDirectDataFeedSPMD(module.get(), "data_snd", "fwd_data");
}

const char* kSelectBasecase = R"(
  HloModule test
  ENTRY computation1 {
    compare_true_data = f32[16] parameter(0)
    compare_false_data = f32[16] parameter(1)
    device_id_constant = u32[] constant($device_id_constant)
    repl_id = u32[] replica-id()

    prd = pred[] compare(repl_id, device_id_constant), direction=$direction
    bcast = pred[] broadcast(prd), dimensions={}
    selected_data = f32[16] select(bcast, compare_true_data, compare_false_data)
    ROOT data_rcv = f32[16] collective-permute(selected_data),
        source_target_pairs=$pairs
  }
)";

TEST_F(CollectiveSelectFolderTest, EqualTrueBranchTransform) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSelectBasecase, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$device_id_constant", "3"},
                             {"$direction", "EQ"},
                             {"$pairs", "{{3,0}}"}}));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->name(), "compare_true_data");
}

TEST_F(CollectiveSelectFolderTest, EqualFalseBranchTransform) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSelectBasecase, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$device_id_constant", "3"},
                             {"$direction", "EQ"},
                             {"$pairs", "{{0,1},{1,2}}"}}));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->name(), "compare_false_data");
}

TEST_F(CollectiveSelectFolderTest, NotEqualFalseBranchTransform) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSelectBasecase, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$device_id_constant", "3"},
                             {"$direction", "NE"},
                             {"$pairs", "{{3,0}}"}}));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->name(), "compare_false_data");
}

TEST_F(CollectiveSelectFolderTest, NotEqualTrueTrueTransform) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSelectBasecase, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$device_id_constant", "3"},
                             {"$direction", "NE"},
                             {"$pairs", "{{0,1},{1,2},{4,5},{5,6}}"}}));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->name(), "compare_true_data");
}

TEST_F(CollectiveSelectFolderTest, CommutativeCompare) {
  const char* kHlo = R"(
  HloModule test
  ENTRY computation1 {
    data_1 = f32[16] parameter(0)
    data_2 = f32[16] parameter(1)
    c3 = u32[] constant(3)
    partition_id = u32[] partition-id()
    predicate = pred[] compare(c3, partition_id), direction=NE
    bcast_predicate = pred[] broadcast(predicate), dimensions={}
    selected_data = f32[16] select(bcast_predicate, data_1, data_2)
    ROOT data_rcv = f32[16] collective-permute(selected_data),
        source_target_pairs={{0,1},{1,2},{4,5},{5,6}}, channel_id=1
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunAndCheckHloRewrite(kHlo, CollectiveSelectFolder(),
                                                /*expect_change=*/true));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CollectivePermute(op::Parameter(0)));
}

TEST_F(CollectiveSelectFolderTest, MoreThanOnePair_NotTransformed) {
  // The cp contains sources 0 and 1, and therefore doesn't match
  // equal(1) and not equal(1)
  TF_ASSERT_OK(RunAndCheckHloRewrite(kSelectBasecase, CollectiveSelectFolder(),
                                     /*expect_change=*/false,
                                     {{"$device_id_constant", "1"},
                                      {"$direction", "EQ"},
                                      {"$pairs", "{{0,1},{1,2}}"}}));

  // The cp contains sources 0 and 1, and therefore doesn't match
  // not_equal(1) and not not_equal(1)
  TF_ASSERT_OK(RunAndCheckHloRewrite(kSelectBasecase, CollectiveSelectFolder(),
                                     /*expect_change=*/false,
                                     {{"$device_id_constant", "1"},
                                      {"$direction", "NE"},
                                      {"$pairs", "{{0,1},{1,2}}"}}));
}

const char* kSelectNoBroadcast = R"(
  HloModule test
  ENTRY computation1 {
    compare_true_data = f32[16] parameter(0)
    compare_false_data = f32[16] parameter(1)
    device_id_constant = u32[] constant($device_id_constant)
    repl_id = u32[] replica-id()

    prd = pred[] compare(repl_id, device_id_constant), direction=$direction
    selected_data = f32[16] select(prd, compare_true_data, compare_false_data)
    ROOT data_rcv = f32[16] collective-permute(selected_data),
        source_target_pairs=$pairs
  }
)";

TEST_F(CollectiveSelectFolderTest, SelectNoBroadcastTransform) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kSelectNoBroadcast, CollectiveSelectFolder(),
                            /*expect_change=*/true,
                            {{"$device_id_constant", "3"},
                             {"$direction", "EQ"},
                             {"$pairs", "{{3,0}}"}}));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->name(), "compare_true_data");
}

TEST_F(CollectiveSelectFolderTest, NegatedPredicate_NotTransformed) {
  const absl::string_view kHlo = R"(
    HloModule test
    ENTRY computation {
      data_1 = f32[16] parameter(0)
      data_2 = f32[16] parameter(1)
      c3 = u32[] constant(3)
      partition_id = u32[] partition-id()
      predicate = pred[] compare(partition_id, c3), direction=EQ
      negated_predicate = pred[] not(predicate)
      selected_data = f32[16] select(negated_predicate, data_1, data_2)
      ROOT result_data = f32[16] collective-permute(selected_data),
          source_target_pairs={{3,0}}, channel_id=1
    }
  )";
  TF_ASSERT_OK(ExpectNoTranform(kHlo));
}

TEST_F(CollectiveSelectFolderTest, ReplicaIdChannelIdMismatch_NotTransformed) {
  const absl::string_view hlo = R"(
    HloModule test
    ENTRY computation1 {
      compare_true_data = f32[16] parameter(0)
      compare_false_data = f32[16] parameter(1)
      device_id_constant = u32[] constant(0)
      repl_id = u32[] replica-id()

      prd = pred[] compare(repl_id, device_id_constant), direction=EQ
      selected_data = f32[16] select(prd, compare_true_data, compare_false_data)
      ROOT data_rcv = f32[16] collective-permute(selected_data), channel_id=1,
          source_target_pairs={{0,1}}
    }
  )";
  TF_ASSERT_OK(ExpectNoTranform(hlo));
}

TEST_F(CollectiveSelectFolderTest, PartIdChannelIdMismatch_NotTransformed) {
  const absl::string_view hlo = R"(
    HloModule test
    ENTRY computation1 {
      compare_true_data = f32[16] parameter(0)
      compare_false_data = f32[16] parameter(1)
      device_id_constant = u32[] constant(0)
      repl_id = u32[] partition-id()

      prd = pred[] compare(repl_id, device_id_constant), direction=EQ
      selected_data = f32[16] select(prd, compare_true_data, compare_false_data)
      ROOT data_rcv = f32[16] collective-permute(selected_data),
          source_target_pairs={{0,1}}
    }
  )";
  TF_ASSERT_OK(ExpectNoTranform(hlo));
}

TEST_F(CollectiveSelectFolderTest, WrongNesting_NotTransformed) {
  const absl::string_view hlo = R"(
    HloModule test
    ENTRY computation1 {
      compare_true_data = f32[16] parameter(0)
      compare_false_data = f32[16] parameter(1)
      device_id_constant = u32[] constant(0)
      repl_id = u32[] replica-id()
      sum = u32[] add(device_id_constant, repl_id)  // additional op

      prd = pred[] compare(sum, device_id_constant), direction=EQ
      selected_data = f32[16] select(prd, compare_true_data, compare_false_data)
      ROOT data_rcv = f32[16] collective-permute(selected_data),
          source_target_pairs={{0,1}}
    }
  )";
  TF_ASSERT_OK(ExpectNoTranform(hlo));
}

// This select predicate operates on `replica-id` while the two collectives are
// in the different partition dimension, so the predicate and collectives reason
// over DIFFERENT ids. We can therefore NOT fold this select away on either
// path.
// Represents real-life usecase.
TEST_F(CollectiveSelectFolderTest,
       CondensedPipelineParallelism_IncompatibleGroupMode) {
  const absl::string_view kHlo = R"(
    HloModule test

    ENTRY computation {
      param = (f32[8192], f32[8192]) parameter(0)
      replica_id = u32[] replica-id()
      c0 = u32[] constant(0)
      c3 = u32[] constant(3)
      operand_predicate = pred[] compare(replica_id, c3), direction=EQ
      result_predicate = pred[] compare(replica_id, c0), direction=EQ
      operand_bwd = f32[8192] get-tuple-element(param), index=0
      operand_fwd = f32[8192] get-tuple-element(param), index=1
      select_operand = f32[8192] select(operand_predicate, operand_bwd,
          operand_fwd)
      cp.backward = f32[8192] collective-permute(select_operand), channel_id=1,
          source_target_pairs={{3,0}}
      cp.forward = f32[8192] collective-permute(select_operand), channel_id=2,
          source_target_pairs={{0,1},{1,2},{2,3}}
      ROOT select_result = f32[8192] select(result_predicate, cp.backward,
          cp.forward)
    }
  )";
  TF_ASSERT_OK(ExpectNoTranform(kHlo));
}

// This select predicate operates on `partition-id` with collectives in the same
// partition dimension, so the predicate and collectives reason over the same
// ids. The tested id, 3, appears as an exclusive source in one of the
// collectives and does not appear as a source in the other collective. We can
// therefore fold the select away on both paths.
// Represents real-life usecase.
TEST_F(CollectiveSelectFolderTest,
       CondensedPipelineParallelism_CompatibleGroupMode) {
  const absl::string_view kHlo = R"(
    HloModule test

    ENTRY computation {
      param = (f32[8192], f32[8192]) parameter(0)
      partition_id = u32[] partition-id()
      c0 = u32[] constant(0)
      c3 = u32[] constant(3)
      operand_predicate = pred[] compare(partition_id, c3), direction=EQ
      result_predicate = pred[] compare(partition_id, c0), direction=EQ
      operand_bwd = f32[8192] get-tuple-element(param), index=0
      operand_fwd = f32[8192] get-tuple-element(param), index=1
      select_operand = f32[8192] select(operand_predicate, operand_bwd,
          operand_fwd)
      cp.backward = f32[8192] collective-permute(select_operand), channel_id=1,
          source_target_pairs={{3,0}}
      cp.forward = f32[8192] collective-permute(select_operand), channel_id=2,
          source_target_pairs={{0,1},{1,2},{2,3}}
      ROOT select_result = f32[8192] select(result_predicate, cp.backward,
          cp.forward)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunAndCheckHloRewrite(kHlo, CollectiveSelectFolder(),
                                                /*expect_change=*/true));
  const absl::string_view kExpected = R"(
    // CHECK:      ENTRY %computation
    // CHECK:        %[[PARAM:.*]] = (f32[8192]{0}, f32[8192]{0}) parameter(0)
    // CHECK:        %[[OPERAND_BWD:.*]] = {{.*}} get-tuple-element
    // CHECK-SAME:       ({{.*}} %[[PARAM]]), index=0
    // CHECK:        %[[OPERAND_FWD:.*]] = {{.*}} get-tuple-element
    // CHECK-SAME:       ({{.*}} %[[PARAM]]), index=1
    // CHECK:        %[[CP_BWD:.*]] = {{.*}} collective-permute
    // CHECK-SAME:       ({{.*}} %[[OPERAND_BWD]]), channel_id=1,
    // CHECK-SAME:       source_target_pairs={{\{}}{3,0}}
    // CHECK:        %[[CP_FWD:.*]] = {{.*}} collective-permute
    // CHECK-SAME:       ({{.*}} %[[OPERAND_FWD]]), channel_id=2,
    // CHECK-SAME:       source_target_pairs={{\{}}{0,1},{1,2},{2,3}}
    // CHECK:        ROOT %{{.*}} =
    // CHECK-SAME:       select({{.*}} %{{.*}}, {{.*}} %[[CP_BWD]],
    // CHECK-SAME:       %[[CP_FWD]])
    // CHECK:      }
  )";
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(CollectiveSelectFolderTest, DtypeConvertedPartitionId) {
  const absl::string_view kHlo = R"(
    HloModule test

    ENTRY computation {
      param = (f32[1,1,28672,2048]{3,2,1,0}, f32[1,1,28672,2048]{3,2,1,0})
          parameter(0)
      get-tuple-element-a = f32[1,1,28672,2048]{3,2,1,0}
          get-tuple-element(param), index=0
      get-tuple-element-b = f32[1,1,28672,2048]{3,2,1,0}
          get-tuple-element(param), index=1
      partition-id.1 = u32[] partition-id()
      convert = s32[] convert(partition-id.1)
      constant.148 = s32[] constant(3)
      compare.83 = pred[] compare(convert, constant.148), direction=EQ
      select.33 = f32[1,1,28672,2048]{3,2,1,0} select(compare.83,
          get-tuple-element-a, get-tuple-element-b)
      ROOT cp-a = f32[1,1,28672,2048]{3,2,1,0} collective-permute(select.33),
          channel_id=1, source_target_pairs={{3,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunAndCheckHloRewrite(kHlo, CollectiveSelectFolder(),
                                                /*expect_change=*/true));
  const absl::string_view kExpected = R"(
    // CHECK: %[[PARAM:.*]] = {{.*}} parameter(0)
    // CHECK: %[[DATA_A:.*]] = {{.*}} get-tuple-element({{.*}} %[[PARAM]]), index=0
    // CHECK: ROOT %[[DATA_A_:.*]] = {{.*}} collective-permute({{.*}} %[[DATA_A]])
  )";
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_result);
}

}  // namespace
}  // namespace xla
