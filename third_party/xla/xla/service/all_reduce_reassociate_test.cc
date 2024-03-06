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

#include "xla/service/all_reduce_reassociate.h"

#include <cstddef>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/statusor.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;
using ::testing::_;

class AllReduceSimplifierTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, bool expect_change,
      bool reassociate_converted_ar = false) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed =
        AllReduceReassociate(reassociate_converted_ar).Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t AllReduceCount(std::unique_ptr<HloModule>& module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<HloOpcode::kAllReduce>);
  }
};

TEST_F(AllReduceSimplifierTest, Simple) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, SimpleWithChannelId) {
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
  ar0 = f32[8] all-reduce(p0), channel_id=1, replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), channel_id=1, replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a linear chain of adds of ARs is reassociated iin a single
// pass.
TEST_F(AllReduceSimplifierTest, SimpleChain) {
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
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[8] all-reduce(p2), replica_groups={}, to_apply=sum
  ar3 = f32[8] all-reduce(p3), replica_groups={}, to_apply=sum
  add0 = f32[8] add(ar0, ar1)
  add1 = f32[8] add(add0, ar2)
  ROOT add2 = f32[8] add(add1, ar3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::AllReduce(m::Add(
          m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
          m::Parameter(3))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a tree of add of ARs is reassociated in a single pass.
TEST_F(AllReduceSimplifierTest, SimpleTree) {
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
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[8] all-reduce(p2), replica_groups={}, to_apply=sum
  ar3 = f32[8] all-reduce(p3), replica_groups={}, to_apply=sum
  add0 = f32[8] add(ar0, ar1)
  add1 = f32[8] add(ar2, ar3)
  ROOT add2 = f32[8] add(add0, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Add(m::Parameter(0), m::Parameter(1)),
                                  m::Add(m::Parameter(2), m::Parameter(3)))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, MismatchOp0) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=max
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchOp1) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=max
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=max
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchReplicaGroups) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={{0}}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchHasChannelId) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, channel_id=3, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchUseGlobalDeviceId) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={{0, 1}}, channel_id=3, use_global_device_ids=true, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={{0, 1}}, channel_id=4, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, NotSingleUser) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  add = f32[8] add(ar0, ar1)
  ROOT t = (f32[8], f32[8]) tuple(ar0, add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, DoubleUse) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  add = f32[8] add(ar0, ar0)
  ROOT c = f32[8] copy(add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
}

TEST_F(AllReduceSimplifierTest, PaddedUse) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  %constant.1 = f32[] constant(0)
  pad = f32[12]{0} pad(ar0, constant.1), padding=0_4
  pad.1 = f32[12]{0} pad(ar1, constant.1), padding=0_4
  ROOT add = f32[12] add(pad, pad.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Pad(m::Parameter(0), _),
                                  m::Pad(m::Parameter(1), _))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, PaddedUseInvalidReduceValue) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  %constant.1 = f32[] constant(-1.0)
  pad = f32[12]{0} pad(ar0, constant.1), padding=0_4
  pad.1 = f32[12]{0} pad(ar1, constant.1), padding=0_4
  ROOT add = f32[12] add(pad, pad.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 2);
}

TEST_F(AllReduceSimplifierTest, PaddedUseNotProfitable) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  %constant.1 = f32[] constant(0)
  pad = f32[17]{0} pad(ar0, constant.1), padding=0_9
  pad.1 = f32[17]{0} pad(ar1, constant.1), padding=0_9
  ROOT add = f32[17] add(pad, pad.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 2);
}

TEST_F(AllReduceSimplifierTest, PaddedUseDoubleUseNotProfitable) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  %constant.1 = f32[] constant(0)
  pad = f32[9]{0} pad(ar0, constant.1), padding=0_1
  ROOT add = f32[9] add(pad, pad)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, ReshapeUse) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[1,8] parameter(0)
  p1 = f32[1,8] parameter(1)
  ar0 = f32[1,8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[1,8] all-reduce(p1), replica_groups={}, to_apply=sum
  rshp0 = f32[8]{0} reshape(ar0)
  rshp1 = f32[8]{0} reshape(ar1)
  ROOT add = f32[8] add(rshp0, rshp1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Reshape(m::Parameter(0)),
                                  m::Reshape(m::Parameter(1)))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, SliceUse) {
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
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  rshp0 = f32[4]{0} slice(ar0), slice={[0:4]}
  rshp1 = f32[4]{0} slice(ar1), slice={[0:4]}
  ROOT add = f32[4] add(rshp0, rshp1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Slice(m::Parameter(0)),
                                  m::Slice(m::Parameter(1)))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a linear chain of converts-adds of ARs is reassociated in a
// single pass.
TEST_F(AllReduceSimplifierTest, ChainWithConvert) {
  absl::string_view hlo_string = R"(
HloModule m
add.1 {
  x.47 = bf16[] parameter(0)
  y.47 = bf16[] parameter(1)
  ROOT add.2532 = bf16[] add(x.47, y.47)
}
ENTRY main {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[8] parameter(2)
  p3 = bf16[8] parameter(3)
  ar0 = bf16[8] all-reduce(p0), replica_groups={}, to_apply=add.1
  ar1 = bf16[8] all-reduce(p1), replica_groups={}, to_apply=add.1
  ar2 = bf16[8] all-reduce(p2), replica_groups={}, to_apply=add.1
  ar3 = bf16[8] all-reduce(p3), replica_groups={}, to_apply=add.1
  convert0 = f32[8] convert(ar0)
  convert1 = f32[8] convert(ar1)
  add0 = f32[8] add(convert0, convert1)
  convert2 = f32[8] convert(ar2)
  add1 = f32[8] add(add0, convert2)
  convert3 = f32[8] convert(ar3)
  add2 = f32[8] add(add1, convert3)
  ROOT convert4 = bf16[8] convert(add2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true,
                                  /*reassociate_converted_ar*/ true));
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::Convert(m::AllReduce(m::Add(m::Add(m::Add(m::Convert(m::Parameter(0)),
                                                   m::Convert(m::Parameter(1))),
                                            m::Convert(m::Parameter(2))),
                                     m::Convert(m::Parameter(3))))));
  EXPECT_EQ(AllReduceCount(module), 1);
  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operand(0)->shape(),
      GmockMatch(::xla::match::Shape().WithElementType(F32)));
}

// Checks that a list of incompatible converts-adds of ARs should NOT be
// reassociated.
TEST_F(AllReduceSimplifierTest, AllreduceWithConvertIncompatibleType) {
  absl::string_view hlo_string = R"(
HloModule m
add.1 {
  x.47 = bf16[] parameter(0)
  y.47 = bf16[] parameter(1)
  ROOT add.2532 = bf16[] add(x.47, y.47)
}
max.1 {
  x.48 = bf16[] parameter(0)
  y.48 = bf16[] parameter(1)
  ROOT max.2533 = bf16[] maximum(x.48, y.48)
}
min.1 {
  x.49 = bf16[] parameter(0)
  y.49 = bf16[] parameter(1)
  ROOT min.2534 = bf16[] minimum(x.49, y.49)
}
mul.1 {
  x.50 = bf16[] parameter(0)
  y.50 = bf16[] parameter(1)
  ROOT mul.2535 = bf16[] multiply(x.50, y.50)
}
ENTRY main {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[8] parameter(2)
  p3 = bf16[8] parameter(3)
  ar0 = bf16[8] all-reduce(p0), replica_groups={}, to_apply=add.1
  ar1 = bf16[8] all-reduce(p1), replica_groups={}, to_apply=max.1
  ar2 = bf16[8] all-reduce(p2), replica_groups={}, to_apply=min.1
  ar3 = bf16[8] all-reduce(p3), replica_groups={}, to_apply=mul.1
  convert0 = f32[8] convert(ar0)
  convert1 = f32[8] convert(ar1)
  add0 = f32[8] add(convert0, convert1)
  convert2 = f32[8] convert(ar2)
  add1 = f32[8] add(add0, convert2)
  convert3 = f32[8] convert(ar3)
  add2 = f32[8] add(add1, convert3)
  ROOT convert4 = bf16[8] convert(add2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
  SCOPED_TRACE(module->ToString());
}

// Checks that a list of incompatible converts-adds of ARs should NOT be
// reassociated.
TEST_F(AllReduceSimplifierTest, AllreduceWithLossyConvert) {
  absl::string_view hlo_string = R"(
HloModule m
add.1 {
  x.47 = bf16[] parameter(0)
  y.47 = bf16[] parameter(1)
  ROOT add.2532 = bf16[] add(x.47, y.47)
}
ENTRY main {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[8] parameter(2)
  p3 = bf16[8] parameter(3)
  ar0 = bf16[8] all-reduce(p0), replica_groups={}, to_apply=add.1
  ar1 = bf16[8] all-reduce(p1), replica_groups={}, to_apply=add.1
  ar2 = bf16[8] all-reduce(p2), replica_groups={}, to_apply=add.1
  ar3 = bf16[8] all-reduce(p3), replica_groups={}, to_apply=add.1
  convert0 = u32[8] convert(ar0)
  convert1 = u32[8] convert(ar1)
  add0 = u32[8] add(convert0, convert1)
  convert2 = u32[8] convert(ar2)
  add1 = u32[8] add(add0, convert2)
  convert3 = u32[8] convert(ar3)
  add2 = u32[8] add(add1, convert3)
  ROOT convert4 = bf16[8] convert(add2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
  SCOPED_TRACE(module->ToString());
}

TEST_F(AllReduceSimplifierTest, AllReduceDynamicSlicePattern) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[1,8] parameter(0)
  p1 = f32[1,8] parameter(1)
  p2 = f32[1,8] parameter(2)
  p3 = s32[] parameter(3)
  cst = s32[] constant(0)
  ar0 = f32[1,8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[1,8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[1,8] all-reduce(p2), replica_groups={}, to_apply=sum
  dyn0 = f32[1,4] dynamic-slice(ar0, cst, p3), dynamic_slice_sizes={1,4}
  dyn1 = f32[1,4] dynamic-slice(ar1, cst, p3), dynamic_slice_sizes={1,4}
  dyn2 = f32[1,4] dynamic-slice(ar2, cst, p3), dynamic_slice_sizes={1,4}
  add = f32[1,4] add(dyn0, dyn1)
  ROOT add1 = f32[1,4] add(add, dyn2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::DynamicSlice(
                  m::AllReduce(m::Add(m::Add(m::Parameter(0), m::Parameter(1)),
                                      m::Parameter(2))),
                  m::Constant(), m::Parameter(3)));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, AllReduceDynamicSlicePatternSameOperand) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[1,8] parameter(0)
  p1 = f32[1,8] parameter(1)
  p2 = s32[] parameter(2)
  cst = s32[] constant(0)
  ar0 = f32[1,8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar2 = f32[1,8] all-reduce(p1), replica_groups={}, to_apply=sum
  dyn0 = f32[1,4] dynamic-slice(ar0, cst, p2), dynamic_slice_sizes={1,4}
  dyn2 = f32[1,4] dynamic-slice(ar2, cst, p2), dynamic_slice_sizes={1,4}
  add = f32[1,4] add(dyn0, dyn0)
  ROOT add1 = f32[1,4] add(add, dyn2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::DynamicSlice(
                  m::AllReduce(m::Add(m::Add(m::Parameter(0), m::Parameter(0)),
                                      m::Parameter(1))),
                  m::Constant(), m::Parameter(2)));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, AllReduceDynamicSliceDifferentSlices) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[1,8] parameter(0)
  p1 = f32[1,8] parameter(1)
  p2 = f32[1,16] parameter(2)
  p3 = s32[] parameter(3)
  cst = s32[] constant(0)
  ar0 = f32[1,8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[1,8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[1,16] all-reduce(p2), replica_groups={}, to_apply=sum
  dyn0 = f32[1,4] dynamic-slice(ar0, cst, p3), dynamic_slice_sizes={1,4}
  dyn1 = f32[1,4] dynamic-slice(ar1, cst, p3), dynamic_slice_sizes={1,4}
  dyn2 = f32[1,4] dynamic-slice(ar2, cst, p3), dynamic_slice_sizes={1,4}
  add = f32[1,4] add(dyn0, dyn1)
  ROOT add1 = f32[1,4] add(add, dyn2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::Add(m::DynamicSlice(),
             m::DynamicSlice(m::AllReduce(), m::Constant(), m::Parameter(3))));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_EQ(AllReduceCount(module), 2);
}

}  // namespace
}  // namespace xla
