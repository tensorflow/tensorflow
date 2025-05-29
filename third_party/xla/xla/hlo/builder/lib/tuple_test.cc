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

#include "xla/hlo/builder/lib/tuple.h"

#include <cstdint>
#include <utility>

#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using TupleTest = ClientLibraryTestRunnerMixin<HloTestBase>;

TEST_F(TupleTest, DisassembleAssemble) {
  XlaBuilder builder(TestName());

  Shape shape = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(S32, {3}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(S32, {4}), ShapeUtil::MakeShape(S32, {5})}),
      ShapeUtil::MakeShape(S32, {6}),
  });
  Literal input = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateFullWithDescendingLayout({3}, int32_t{42}),
      LiteralUtil::MakeTupleOwned(
          LiteralUtil::CreateFullWithDescendingLayout({4}, int32_t{43}),
          LiteralUtil::CreateFullWithDescendingLayout({5}, int32_t{44})),
      LiteralUtil::CreateFullWithDescendingLayout({6}, int32_t{45}));

  XlaOp param = Parameter(&builder, 0, shape, "param");
  TF_ASSERT_OK_AND_ASSIGN(ShapeTree<XlaOp> disassembled_tuple,
                          DisassembleTuple(param));
  int32_t addend = 1;
  disassembled_tuple.ForEachMutableElement([&](const ShapeIndex& index,
                                               XlaOp* element) {
    const Shape& subshape = ShapeUtil::GetSubshape(shape, index);
    if (subshape.IsArray()) {
      *element = Add(
          *element,
          ConstantLiteral(&builder, LiteralUtil::CreateFullWithDescendingLayout(
                                        subshape.dimensions(), addend)));
      ++addend;
    }
  });
  AssembleTuple(&builder, std::move(disassembled_tuple));

  const Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateFullWithDescendingLayout({3}, int32_t{43}),
      LiteralUtil::MakeTupleOwned(
          LiteralUtil::CreateFullWithDescendingLayout({4}, int32_t{45}),
          LiteralUtil::CreateFullWithDescendingLayout({5}, int32_t{47})),
      LiteralUtil::CreateFullWithDescendingLayout({6}, int32_t{49}));
  ComputeAndCompareLiteral(&builder, expected, {&input}, ErrorSpec(0), &shape);
}

}  // namespace
}  // namespace xla
