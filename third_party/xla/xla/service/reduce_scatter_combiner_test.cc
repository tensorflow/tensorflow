/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/reduce_scatter_combiner.h"

#include <cstddef>
#include <utility>

#include <gmock/gmock.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

constexpr int64_t kMaxCombineCount = 256;
constexpr int64_t kMaxByteCount = 10 * 1024 * 1024;

class ReduceScatterCombinerTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, bool expect_change,
      int64_t byte_threshold = kMaxByteCount,
      int64_t count_threshold = kMaxCombineCount, bool combine_by_dim = true,
      bool combine_while_loops = true) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));

    VLOG(1) << "Before running ReduceScatterCombiner: "
            << ReduceScatterCount(module.get()) << " reduce-scatter ops";

    auto changed = ReduceScatterCombiner(byte_threshold, count_threshold,
                                         combine_by_dim, combine_while_loops)
                       .Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }

    VLOG(1) << "After running ReduceScatterCombiner: "
            << ReduceScatterCount(module.get()) << " reduce-scatter ops";

    EXPECT_EQ(changed.value(), expect_change);
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t ReduceScatterCount(HloModule *module) {
    int64_t sum = 0;
    for (auto comp : module->computations()) {
      sum += absl::c_count_if(comp->instructions(),
                              HloPredicateIsOp<HloOpcode::kReduceScatter>);
    }
    return sum;
  }
};

TEST_F(ReduceScatterCombinerTest, Simple) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[4], f32[4]) tuple(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(ReduceScatterCount(module.get()), 1);
}

TEST_F(ReduceScatterCombinerTest, SimpleMultipleGroups) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  p1 = f32[8, 8] parameter(1)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4, 8] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs2 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  rs3 = f32[8, 4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  ROOT t = (f32[4, 8], f32[4, 8], f32[8, 4], f32[8, 4])
      tuple(rs0, rs1, rs2, rs3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(ReduceScatterCount(module.get()), 2);
}

TEST_F(ReduceScatterCombinerTest, DifferentDimensions) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  p1 = f32[8, 8] parameter(1)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4, 8] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs2 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  rs3 = f32[8, 4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  ROOT t = (f32[4, 8], f32[4, 8], f32[8, 4], f32[8, 4])
      tuple(rs0, rs1, rs2, rs3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, RunPass(hlo_string, /*expect_change=*/true, kMaxByteCount,
                           kMaxCombineCount, /*combine_by_dim=*/false));
  EXPECT_EQ(ReduceScatterCount(module.get()), 1);
}

TEST_F(ReduceScatterCombinerTest, DifferentDimensionsAndRanks) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  rs1 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  rs2 = f32[4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[8, 4], f32[8, 4], f32[4])
      tuple(rs0, rs1, rs2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, RunPass(hlo_string, /*expect_change=*/true, kMaxByteCount,
                           kMaxCombineCount, /*combine_by_dim=*/false));
  EXPECT_EQ(ReduceScatterCount(module.get()), 1);
}

// Test that dependent reduce-scatter do not get combined.
TEST_F(ReduceScatterCombinerTest, DependentReduceScatter) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[2, 8] reduce-scatter(rs0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[4, 8], f32[2, 8]) tuple(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterCombinerTest, DoNotCombineMismatched) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), replica_groups={{1,0}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[4], f32[4]) tuple(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterCombinerTest, DoNotCombineWithoutReductionKind) {
  absl::string_view hlo_string = R"(
HloModule TestModule

region_0 {
  Arg_1 = bf16[] parameter(1)
  Arg_0 = bf16[] parameter(0)
  convert_1 = f32[] convert(Arg_1)
  convert_0 = f32[] convert(Arg_0)
  add0 = f32[] add(convert_1, convert_0)
  ROOT convert_2 = bf16[] convert(add0)
}

region_1 {
  Arg_1 = bf16[] parameter(1)
  Arg_0 = bf16[] parameter(0)
  convert_1 = f32[] convert(Arg_1)
  convert_0 = f32[] convert(Arg_0)
  add0 = f32[] add(convert_1, convert_0)
  ROOT convert_2 = bf16[] convert(add0)
}

ENTRY entry{
  param0 = bf16[512,256]{1,0} parameter(0)
  param1 = bf16[512,256]{1,0} parameter(1)
  reduce-scatter.0 = bf16[512,256]{1,0} reduce-scatter(param0),
      replica_groups={{0}}, dimensions={0}, to_apply=region_0
  reduce-scatter.1 = bf16[512,256]{1,0} reduce-scatter(param1),
      replica_groups={{0}}, dimensions={0}, to_apply=region_1
  ROOT add.0 = tuple(reduce-scatter.0, reduce-scatter.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterCombinerTest, HighThreshold) {
  absl::string_view hlo_string = R"(
HloModule m

sum_reduce {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

ENTRY main {
  param.0 = bf16[1024,32768]{1,0} parameter(0)
  param.1 = bf16[4096,8192]{1,0} parameter(1)
  param.2 = bf16[3,128,64,1024]{2,1,0,3}parameter(2)
  param.3 = bf16[1024,128,64]{2,1,0} parameter(3)
  reduce-scatter.19 = bf16[1024,32768]{1,0} reduce-scatter(param.0),
      channel_id=132, replica_groups={{0}}, dimensions={0}, to_apply=sum_reduce
  reduce-scatter.21 = bf16[4096,8192]{1,0} reduce-scatter(param.1),
      channel_id=134, replica_groups={{0}}, dimensions={0}, to_apply=sum_reduce
  reduce-scatter.23 = bf16[3,128,64,1024]{2,1,0,3} reduce-scatter(param.2),
      channel_id=136, replica_groups={{0}}, dimensions={3}, to_apply=sum_reduce
  reduce-scatter.25 = bf16[1024,128,64]{2,1,0} reduce-scatter(param.3),
      channel_id=138, replica_groups={{0}}, dimensions={0}, to_apply=sum_reduce
  ROOT tuple = tuple(reduce-scatter.19, reduce-scatter.21, reduce-scatter.23,
      reduce-scatter.25)
})";

  int64_t combined_bytes = 67108864 + 67108864 + 50331648 + 16777216;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      RunPass(hlo_string, /*expect_change=*/true,
              /*byte_threshold=*/combined_bytes,
              /*count_threshold=*/kMaxCombineCount, /*combine_by_dim=*/false));
  EXPECT_EQ(ReduceScatterCount(module.get()), 1);
}

TEST_F(ReduceScatterCombinerTest, DoNotCombineInWhileLoop) {
  absl::string_view hlo_string = R"(
HloModule m

sum_reduce {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_body {
  param = (bf16[1024,32768]{1,0}, bf16[4096,8192]{1,0}) parameter(0)
  param.0 = bf16[1024,32768]{1,0} get-tuple-element(param), index=0
  param.1 = bf16[4096,8192]{1,0} get-tuple-element(param), index=1
  reduce-scatter.0 = bf16[1024,32768]{1,0} reduce-scatter(param.0),
      channel_id=132, replica_groups={{0}}, dimensions={0}, to_apply=sum_reduce
  reduce-scatter.1 = bf16[4096,8192]{1,0} reduce-scatter(param.1),
      channel_id=134, replica_groups={{0}}, dimensions={0}, to_apply=sum_reduce
  ROOT tuple = tuple(reduce-scatter.0, reduce-scatter.1)
}

while_cond {
  param = (bf16[1024,32768], bf16[4096,8192]) parameter(0)
  ROOT cond = pred[] constant(true)
}

ENTRY main {
  param.0 = bf16[1024,32768]{1,0} parameter(0)
  param.1 = bf16[4096,8192]{1,0} parameter(1)
  while_init = (bf16[1024,32768], bf16[4096,8192]) tuple(param.0, param.1)
  while_loop = (bf16[1024,32768], bf16[4096,8192]) while(while_init), condition=while_cond, body=while_body
  gte.0 = bf16[1024,32768] get-tuple-element(while_loop), index=0
  gte.1 = bf16[4096,8192] get-tuple-element(while_loop), index=1
  ROOT tuple = tuple(gte.0, gte.1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ReduceScatterCombiner combine(1024 * 1024, kMaxCombineCount,
                                /*combine_by_dim=*/false,
                                /*combine_while_loops=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReduceScatterCombinerTest, PreservesMetadata) {
  absl::string_view hlo_string = R"(
    HloModule Module

    %add (x: f32[], y: f32[]) -> f32[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %x, f32[] %y)
    }

    ENTRY entry {
      %param.0 = f32[32] parameter(0)
      %param.1 = f32[32] parameter(1)
      %rs.0 = f32[16] reduce-scatter(%param.0), replica_groups={{0,1}}, dimensions={0}, to_apply=%add, metadata={op_type="test_type0" op_name="test_name0"}
      %rs.1 = f32[16] reduce-scatter(%param.1), replica_groups={{0,1}}, dimensions={0}, to_apply=%add, metadata={op_type="test_type1" op_name="test_name1"}
      ROOT tuple = (f32[16], f32[16]) tuple(%rs.0, %rs.1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  OpMetadata metadata;
  metadata.set_op_type("test_type0");
  metadata.set_op_name("test_name0");
  auto combined_reduce_scatter = op::Metadata(metadata);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_reduce_scatter, 0),
                        op::GetTupleElement(combined_reduce_scatter, 1)));
}

}  // namespace
}  // namespace xla
