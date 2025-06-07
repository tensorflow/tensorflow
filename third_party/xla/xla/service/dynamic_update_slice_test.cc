/* Copyright 2019 The OpenXLA Authors.

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

#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using DynamicUpdateSliceTest =
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(DynamicUpdateSliceTest, ShardedInPlaceDUS) {
  // A dynamic-update-slice within a while loop.  This construction is an easy
  // way to make a DUS which can be run "in-place" (i.e. the input and output
  // are the same buffer, and running the DUS only writes to the updated
  // elements).
  const char kModuleStr[] = R"(
  HloModule test

  body {
    zero = s32[] constant(0)
    one = s32[] constant(1)
    ten = s32[] constant(10)
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    i = s32[] get-tuple-element(loop_carry), index=0
    i_plus_ten = s32[] add(i, ten)
    update = u32[1,100] get-tuple-element(loop_carry), index=1
    data = u32[10000,100] get-tuple-element(loop_carry), index=2
    new_data = u32[10000,100] dynamic-update-slice(data, update, i_plus_ten, zero)
    new_i = s32[] add(i, one)
    ROOT tuple = (s32[], u32[1,100], u32[10000,100]) tuple(new_i, update, new_data)
  }

  cond {
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    two = s32[] constant(2)
    i = s32[] get-tuple-element(loop_carry), index=0
    ROOT less-than = pred[] compare(i, two), direction=LT
  }

  ENTRY test {
    zero = s32[] constant(0)
    initial_i = s32[] parameter(0)
    update = u32[1,100] parameter(1)
    data = u32[10000,100] parameter(2)
    tuple = (s32[], u32[1,100], u32[10000,100]) tuple(initial_i, update, data)
    ROOT while = (s32[], u32[1,100], u32[10000,100]) while(tuple), condition=cond, body=body
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  TF_ASSERT_OK_AND_ASSIGN(auto fake_arguments, MakeFakeArguments(module.get()));
  fake_arguments[0] = LiteralUtil::CreateR0<int32_t>(0);

  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return &const_cast<Literal&>(literal); });

  ErrorSpec no_error(0, 0);
  EXPECT_TRUE(RunAndCompare(std::move(module), fake_argument_ptrs, no_error));
}

// Regression test for a dynamic-update-slice involved in the expansion of a
// kScatter op.  Apologies for the large testcase, this proved difficult to
// reduce.  The bug we're checking for occurs when the dynamic-update-slice is
// run in place but is sharded across cores by ParallelTaskAssigner.
TEST_F(DynamicUpdateSliceTest, ExpandedScatter) {
  const char kModuleStr[] = R"(
HloModule TensorFlowScatter

and.reduce_sub_computation {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT and = pred[] and(lhs, rhs)
}

while_body {
  param.1 = (s32[], f32[8,3,96,1,64]{4,3,2,1,0}, s32[16,4]{1,0}, f32[16,64]{1,0}) parameter(0)
  get-tuple-element.1 = s32[] get-tuple-element(param.1), index=0
  constant.4 = s32[] constant(1)
  add = s32[] add(get-tuple-element.1, constant.4)
  get-tuple-element.2 = f32[8,3,96,1,64]{4,3,2,1,0} get-tuple-element(param.1), index=1
  constant.8 = s32[] constant(0)
  broadcast.1 = s32[5]{0} broadcast(constant.8), dimensions={}
  get-tuple-element.3 = s32[16,4]{1,0} get-tuple-element(param.1), index=2
  constant.5 = s32[] constant(0)
  dynamic-slice = s32[1,4]{1,0} dynamic-slice(get-tuple-element.3, get-tuple-element.1, constant.5), dynamic_slice_sizes={1,4}
  slice.18 = s32[1,1]{1,0} slice(dynamic-slice), slice={[0:1], [0:1]}
  reshape.23 = s32[1]{0} reshape(slice.18)
  reshape.4 = s32[4]{0} reshape(dynamic-slice)
  slice.19 = s32[3]{0} slice(reshape.4), slice={[1:4]}
  constant.6 = s32[1]{0} constant({0})
  concatenate.1 = s32[5]{0} concatenate(reshape.23, slice.19, constant.6), dimensions={0}
  compare.1 = pred[5]{0} compare(broadcast.1, concatenate.1), direction=LE
  constant.9 = s32[5]{0} constant({7, 2, 95, 0, 0})
  compare.2 = pred[5]{0} compare(constant.9, concatenate.1), direction=GE
  and.1 = pred[5]{0} and(compare.1, compare.2)
  constant.10 = pred[] constant(true)
  reduce = pred[] reduce(and.1, constant.10), dimensions={0}, to_apply=and.reduce_sub_computation
  broadcast.2 = pred[1,1,1,1,64]{4,3,2,1,0} broadcast(reduce), dimensions={}
  reshape.24 = s32[] reshape(slice.18)
  slice.26 = s32[1]{0} slice(reshape.4), slice={[1:2]}
  reshape.10 = s32[] reshape(slice.26)
  slice.27 = s32[1]{0} slice(reshape.4), slice={[2:3]}
  reshape.11 = s32[] reshape(slice.27)
  slice.28 = s32[1]{0} slice(reshape.4), slice={[3:4]}
  reshape.12 = s32[] reshape(slice.28)
  reshape.13 = s32[] reshape(constant.6)
  dynamic-slice.2 = f32[1,1,1,1,64]{4,3,2,1,0} dynamic-slice(get-tuple-element.2, reshape.24, reshape.10, reshape.11, reshape.12, reshape.13), dynamic_slice_sizes={1,1,1,1,64}
  get-tuple-element.4 = f32[16,64]{1,0} get-tuple-element(param.1), index=3
  constant.7 = s32[] constant(0)
  dynamic-slice.1 = f32[1,64]{1,0} dynamic-slice(get-tuple-element.4, get-tuple-element.1, constant.7), dynamic_slice_sizes={1,64}
  reshape.28 = f32[1,1,1,1,64]{4,3,2,1,0} reshape(dynamic-slice.1)
  add.1 = f32[1,1,1,1,64]{4,3,2,1,0} add(dynamic-slice.2, reshape.28)
  select = f32[1,1,1,1,64]{4,3,2,1,0} select(broadcast.2, add.1, dynamic-slice.2)
  reshape.29 = s32[] reshape(slice.18)
  slice.29 = s32[1]{0} slice(reshape.4), slice={[1:2]}
  reshape.15 = s32[] reshape(slice.29)
  slice.30 = s32[1]{0} slice(reshape.4), slice={[2:3]}
  reshape.16 = s32[] reshape(slice.30)
  slice.31 = s32[1]{0} slice(reshape.4), slice={[3:4]}
  reshape.17 = s32[] reshape(slice.31)
  reshape.18 = s32[] reshape(constant.6)
  dynamic-update-slice = f32[8,3,96,1,64]{4,3,2,1,0} dynamic-update-slice(get-tuple-element.2, select, reshape.29, reshape.15, reshape.16, reshape.17, reshape.18)
  ROOT tuple.1 = (s32[], f32[8,3,96,1,64]{4,3,2,1,0}, s32[16,4]{1,0}, f32[16,64]{1,0}) tuple(add, dynamic-update-slice, get-tuple-element.3, get-tuple-element.4)
}

while_cond {
  param.0 = (s32[], f32[8,3,96,1,64]{4,3,2,1,0}, s32[16,4]{1,0}, f32[16,64]{1,0}) parameter(0)
  get-tuple-element = s32[] get-tuple-element(param.0), index=0
  constant.2 = s32[] constant(16)
  ROOT compare = pred[] compare(get-tuple-element, constant.2), direction=LT
}

ENTRY main {
  constant = s32[] constant(0)
  z = f32[] constant(0)
  b = f32[8,3,96,1,64]{4,3,2,1,0} broadcast(z), dimensions={}
  i = s32[8,2,4]{2,1,0} parameter(0)
  reshape = s32[16,4]{1,0} reshape(i)
  u = f32[8,2,64]{2,1,0} parameter(1)
  reshape.1 = f32[16,64]{1,0} reshape(u)
  tuple = (s32[], f32[8,3,96,1,64]{4,3,2,1,0}, s32[16,4]{1,0}, f32[16,64]{1,0}) tuple(constant, b, reshape, reshape.1)
  while = (s32[], f32[8,3,96,1,64]{4,3,2,1,0}, s32[16,4]{1,0}, f32[16,64]{1,0}) while(tuple), condition=while_cond, body=while_body
  ROOT get-tuple-element.5 = f32[8,3,96,1,64]{4,3,2,1,0} get-tuple-element(while), index=1
}
)";

  Literal updates =
      Literal::CreateFromShape(ShapeUtil::MakeShape(F32, {8, 2, 64}));
  updates.PopulateWithValue(1.0f);

  Literal indices =
      Literal::CreateFromShape(ShapeUtil::MakeShape(S32, {8, 2, 4}));
  indices
      .Populate<int>([&](absl::Span<const int64_t> indices) -> int {
        auto i = indices[2] + indices[1] * 4 + indices[0] * 2 * 4;
        switch (indices[2]) {
          case 0:
            return i % 8;
          case 1:
            return i % 3;
          case 2:
            return i % 96;
          default:
            return 0;
        }
      })
      .IgnoreError();

  ErrorSpec no_error(0, 0);
  EXPECT_TRUE(RunAndCompare(ParseAndReturnVerifiedModule(kModuleStr).value(),
                            {&indices, &updates}, no_error));
}

}  // anonymous namespace
}  // namespace xla
