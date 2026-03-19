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

#include <memory>
#include <utility>
#include <vector>

#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using DeconstructTupleTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

TEST_F(DeconstructTupleTest, DeconstructTuple) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, ExecuteAndTransfer(&builder, {}));

  std::vector<Literal> elements = result.DecomposeTuple();
  ASSERT_EQ(elements.size(), 2);

  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements[0]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, elements[1]);
}

TEST_F(DeconstructTupleTest, DeconstructTupleTwice) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, ExecuteAndTransfer(&builder, {}));

  // Create a copy of the literal since DecomposeTuple consumes it.
  Literal result_copy = result.Clone();

  std::vector<Literal> elements1 = result.DecomposeTuple();
  std::vector<Literal> elements2 = result_copy.DecomposeTuple();

  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements1[0]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, elements1[1]);
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements2[0]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, elements2[1]);
}

TEST_F(DeconstructTupleTest, DeconstructTupleRepeatedElement) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2, const2, const1});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, ExecuteAndTransfer(&builder, {}));

  std::vector<Literal> elements = result.DecomposeTuple();
  ASSERT_EQ(elements.size(), 4);

  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements[0]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, elements[1]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, elements[2]);
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements[3]);
}

TEST_F(DeconstructTupleTest, DeconstructTupleFromParam) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR1<float>({3.14f, -100.25f});
  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "param0");
  Tuple(&builder, {p});

  const Literal* args[] = {&param0_literal};
  TF_ASSERT_OK_AND_ASSIGN(Literal result, ExecuteAndTransfer(&builder, args));

  std::vector<Literal> elements = result.DecomposeTuple();
  ASSERT_EQ(elements.size(), 1);
  LiteralTestUtil::ExpectR1Equal<float>({3.14f, -100.25f}, elements[0]);
}

TEST_F(DeconstructTupleTest, DeconstructNestedTuple) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2, const1});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, ExecuteAndTransfer(&builder, {}));

  // Literal::DecomposeTuple just unzips one layer.
  std::vector<Literal> elements = result.DecomposeTuple();
  ASSERT_EQ(elements.size(), 3);
  ASSERT_FALSE(elements[0].shape().IsTuple());
  ASSERT_FALSE(elements[1].shape().IsTuple());
  ASSERT_FALSE(elements[2].shape().IsTuple());

  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements[0]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, elements[1]);
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, elements[2]);
}

}  // namespace
}  // namespace xla
