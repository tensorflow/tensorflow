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

#include "xla/service/gpu/transforms/collective_permute_cycle_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using CollectivePermuteCycleDecomposerTest = HloTestBase;

TEST_F(CollectivePermuteCycleDecomposerTest, TrivialNotTransformed) {
  const absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      p = u32[8,8] parameter(0)
      ROOT start = u32[8,8] collective-permute(p), channel_id=1,
        source_target_pairs={{0,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteCycleDecomposerTest, BelowThresholdNotTransformed) {
  // When `size of data` > `threshold`, then it is decomposed, otherwise it
  // stays as it is.
  const absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      p = u32[4,2] parameter(0)
      ROOT start = u32[4,2] collective-permute(p), channel_id=1,
        source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/33);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(CollectivePermuteCycleDecomposer(33), module.get()));
  EXPECT_FALSE(changed);
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(CollectivePermuteCycleDecomposer(16), module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(CollectivePermuteCycleDecomposerTest, ForwardCycle) {
  // For a forward cycle, this test checks:
  // 1. Split collectives should hand channel ids.
  // 2. They should split over the value of partition-id.
  // 3. The metadata and frontend_attributes are propagated to split
  // collectives.
  const absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      p = u32[8,8] parameter(0)
      ROOT start = u32[8,8] collective-permute(p), channel_id=1,
        source_target_pairs={{0,1},{1,2},{2,3},{3,0}},
        frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"},
        metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_CHECK_OK(VerifyHloModule(module.get(), false, true));
  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_include_layout_in_shapes(false);
  EXPECT_TRUE(*RunFileCheck(module->ToString(options), R"(
    // CHECK:     ENTRY %test_computation (p: u32[8,8]) -> u32[8,8] {
    // CHECK-DAG:   %[[partition_id:.+]] = u32[] partition-id()
    // CHECK-DAG:   %[[c0:.+]] = u32[] constant(0)
    // CHECK-DAG:   %[[compare:.+]] = pred[] compare(%[[partition_id]], %[[c0]]), direction=EQ
    // CHECK-DAG:   %{{.+}} = u32[8,8] parameter(0)
    
    // CHECK-DAG:   %[[cp1:.+]] = u32[8,8] collective-permute(%{{.+}}), channel_id=1, 
    // CHECK-SAME{LITERAL}: source_target_pairs={{3,0}}, frontend_attributes={_xla_send_recv_validation={{3,10}}}, metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
    
    // CHECK-DAG:   %[[cp2:.+]] = u32[8,8] collective-permute(%{{.+}}), channel_id=2, 
    // CHECK-SAME{LITERAL}: source_target_pairs={{0,1},{1,2},{2,3}}, frontend_attributes={_xla_send_recv_validation={{0,7},{1,8},{2,9}}}, metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
    
    // CHECK-DAG:   ROOT %select = u32[8,8] select(%[[compare]], %[[cp1]], %[[cp2]])
    // CHECK-DAG: }
  )"));
}

TEST_F(CollectivePermuteCycleDecomposerTest, ForwardCycleNoChannel) {
  // For a forward cycle, this checks:
  // 1. Split collectives should not have channel-id
  // 2. Split collectives are combined based on replica-id.
  const absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      p = u32[8,8] parameter(0)
      ROOT start = u32[8,8] collective-permute(p),
        source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_CHECK_OK(VerifyHloModule(module.get(), false, true));

  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_include_layout_in_shapes(false);
  EXPECT_TRUE(*RunFileCheck(module->ToString(options), R"(
    // CHECK:     ENTRY %test_computation (p: u32[8,8]) -> u32[8,8] {
    // CHECK-DAG:   %[[replica_id:.+]] = u32[] replica-id()
    // CHECK-DAG:   %[[c0:.+]] = u32[] constant(0)
    // CHECK-DAG:   %[[compare:.+]] = pred[] compare(%[[replica_id]], %[[c0]]), direction=EQ
    // CHECK-DAG:   %{{.+}} = u32[8,8] parameter(0)

    // CHECK-DAG:   %[[cp1:.+]] = u32[8,8] collective-permute(%{{.+}}), source_target_pairs=
    // CHECK-SAME{LITERAL}: {{3,0}}
    
    // CHECK-DAG:   %[[cp2:.+]] = u32[8,8] collective-permute(%{{.+}}), source_target_pairs=
    // CHECK-SAME{LITERAL}: {{0,1},{1,2},{2,3}}
    
    // CHECK-DAG:   ROOT %select = u32[8,8] select(%[[compare]], %[[cp1]], %[[cp2]])
    // CHECK-DAG: }
  )"));
}

TEST_F(CollectivePermuteCycleDecomposerTest, ForwardCycleWithMatmul) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  while_cond {
    param = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(param), index=0
    max_iter = u32[] constant(3)
    ROOT cmp = pred[] compare(iter, max_iter), direction=LT
  }

  while_body {
    param = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(param), index=0
    data = f32[2,2] get-tuple-element(param), index=1
    weights = f32[2,2] get-tuple-element(param), index=2
    cp = f32[2,2] collective-permute(data),
      channel_id=1,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}},
      frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"}
    matmul = f32[2,2] dot(weights, cp), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)
  }

  ENTRY test_computation {
    iter = u32[] constant(0)
    data = f32[2,2] parameter(0)
    weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(iter, data, weights)
    while_res = (u32[], f32[2,2], f32[2,2]) while(input), condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_res), index=1
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloCollectivePermuteInstruction* cp1 =
      DynCast<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), "cp.backward"));
  HloCollectivePermuteInstruction* cp2 =
      DynCast<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), "cp.forward"));
  EXPECT_THAT(cp1->ToString(), HasSubstr("source_target_pairs={{3,0}}"));
  EXPECT_THAT(cp1->ToString(), HasSubstr("_xla_send_recv_validation={{3,10}}"));
  EXPECT_THAT(cp2->ToString(),
              HasSubstr("source_target_pairs={{0,1},{1,2},{2,3}}"));
  EXPECT_THAT(cp2->ToString(),
              HasSubstr("_xla_send_recv_validation={{0,7},{1,8},{2,9}}"));
}

TEST_F(CollectivePermuteCycleDecomposerTest, BackwardCycle) {
  // Tests the following for backward cycle:
  // 1. Metadata is propagated to split collectives.
  // 2. Frontend attributes are accurately split.
  // 3. The split collectives have channel IDs.
  const absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      p = u32[8,8] parameter(0)
      ROOT start = u32[8,8] collective-permute(p), channel_id=1,
        source_target_pairs={{0,3},{1,0},{2,1},{3,2}},
        frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"},
        metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(CollectivePermuteCycleDecomposer(0), module.get()));
  EXPECT_TRUE(changed);
  TF_CHECK_OK(VerifyHloModule(module.get(), true, false));
  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_include_layout_in_shapes(false);
  EXPECT_TRUE(*RunFileCheck(module->ToString(options), R"(
    // CHECK:     ENTRY %test_computation (p: u32[8,8]) -> u32[8,8] {
    // CHECK-DAG:   %[[partition:.+]] = u32[] partition-id()
    // CHECK-DAG:   %[[three:.+]] = u32[] constant(3)
    // CHECK-DAG:   %[[compare:.+]] = pred[] compare(%[[partition]], %[[three]]), direction=EQ
    // CHECK-DAG:   %{{.+}} = u32[8,8] parameter(0)
    
    // CHECK-DAG:   %[[cp1:.+]] = u32[8,8] collective-permute(%{{.+}}), channel_id=1, source_target_pairs=
    // CHECK-SAME{LITERAL}: {{0,3}}, frontend_attributes={_xla_send_recv_validation={{0,7}}}, metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
    
    // CHECK-DAG:   %[[cp2:.+]] = u32[8,8] collective-permute(%{{.+}}), channel_id=2, source_target_pairs=
    // CHECK-SAME{LITERAL}: {{1,0},{2,1},{3,2}}, frontend_attributes={_xla_send_recv_validation={{1,8},{2,9},{3,10}}}, metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
    
    // CHECK-DAG:   ROOT %{{.+}} = u32[8,8] select(%[[compare]], %[[cp1]], %[[cp2]])
    // CHECK-DAG: }
  )"));
}

TEST_F(CollectivePermuteCycleDecomposerTest, BackwardCycleNoChannel) {
  // For backward cycle, this checks:
  // 1. Split collectives do not have a channel-id
  // 2. Split collectives are combined based on the value of replica-id.
  const absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      p = u32[8,8] parameter(0)
      ROOT start = u32[8,8] collective-permute(p),
        source_target_pairs={{0,3},{1,0},{2,1},{3,2}},
        frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_include_layout_in_shapes(false);
  TF_CHECK_OK(VerifyHloModule(module.get(), false, true));
  EXPECT_TRUE(*RunFileCheck(module->ToString(options), R"(
    // CHECK:     ENTRY %test_computation (p: u32[8,8]) -> u32[8,8] {
    // CHECK-DAG:   %[[replica_id:.+]] = u32[] replica-id()
    // CHECK-DAG:   %[[three:.+]] = u32[] constant(3)
    // CHECK-DAG:   %[[compare:.+]] = pred[] compare(%[[replica_id]], %[[three]]), direction=EQ
    // CHECK-DAG:   %{{.+}} = u32[8,8] parameter(0)
  
    // CHECK-DAG:   %[[cp1:.+]] = u32[8,8] collective-permute(%{{.+}}), source_target_pairs=
    // CHECK-SAME{LITERAL}: {{0,3}}, frontend_attributes={_xla_send_recv_validation={{0,7}}}
  
    // CHECK-DAG:   %[[cp2:.+]] = u32[8,8] collective-permute(%{{.+}}), source_target_pairs=
    // CHECK-SAME{LITERAL}: {{1,0},{2,1},{3,2}}, frontend_attributes={_xla_send_recv_validation={{1,8},{2,9},{3,10}}}
  
    // CHECK-DAG:   ROOT %select = u32[8,8] select(%[[compare]], %[[cp1]], %[[cp2]])
    // CHECK-DAG: }
  )"));
}

}  // namespace
}  // namespace xla
