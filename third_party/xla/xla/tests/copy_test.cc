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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class CopyOpTest : public HloPjRtTestBase {
 protected:
  CopyOpTest() : platform_(*PlatformUtil::GetDefaultPlatform()) {}

  void TestCopyOp(const Literal& literal) {
    auto builder = HloComputation::Builder(TestName());
    auto constant =
        builder.AddInstruction(HloInstruction::CreateConstant(literal.Clone()));
    builder.AddInstruction(HloInstruction::CreateUnary(
        constant->shape(), HloOpcode::kCopy, constant));
    auto computation = builder.Build();
    auto module = CreateNewVerifiedModule();
    module->AddEntryComputation(std::move(computation));

    Literal result = ExecuteAndTransfer(std::move(module), {});
    EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
  }

  // TODO(vsytch): Remove special handling for dynamic shapes once *all* of XLA
  // supports those as module inputs/outputs.
  void TestDynamicCopyOp(const Literal& literal, const Shape& bounded_shape) {
    Literal dynamic_literal = literal.ToBoundedDynamic(bounded_shape);
    auto builder = HloComputation::Builder(TestName());
    auto parameter = builder.AddInstruction(
        HloInstruction::CreateParameter(0, dynamic_literal.shape(), "param"));
    builder.AddInstruction(HloInstruction::CreateUnary(
        parameter->shape(), HloOpcode::kCopy, parameter));
    auto computation = builder.Build();
    auto module = CreateNewVerifiedModule();
    module->AddEntryComputation(std::move(computation));

    std::vector<Literal*> args = {&dynamic_literal};
    Literal result = ExecuteAndTransfer(std::move(module), args);
    Literal dynamic_result = result.ToBoundedDynamic(bounded_shape);
    EXPECT_TRUE(LiteralTestUtil::Equal(dynamic_literal, dynamic_result));
  }

  void TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3);
  void TestCopyConstantLayoutR4(size_t n1, size_t n2, size_t n3, size_t n4,
                                absl::Span<const int64_t> permutation);

  se::Platform* platform() const { return platform_; }

 private:
  se::Platform* platform_ = nullptr;
};

XLA_TEST_F(CopyOpTest, CopyR0Bool) {
  TestCopyOp(LiteralUtil::CreateR0<bool>(true));
}

XLA_TEST_F(CopyOpTest, CopyR1S0U32) {
  TestCopyOp(LiteralUtil::CreateR1<uint32_t>({}));
}

XLA_TEST_F(CopyOpTest, CopyR1S3U32) {
  TestCopyOp(LiteralUtil::CreateR1<uint32_t>({1, 2, 3}));
}

XLA_TEST_F(CopyOpTest, CopyDynamicR1S1310720U32Dynamic0) {
  // TODO(vsytch): CPU emitter doesn't handle dynamic shapes.
  if (platform()->Name() == "Host") {
    GTEST_SKIP();
  }
  Shape bounded_shape =
      ShapeUtil::MakeShape(PrimitiveType::F32, {1310720}, {true});
  TestDynamicCopyOp(LiteralUtil::CreateRandomLiteral<PrimitiveType::F32>(
                        ShapeUtil::MakeShape(PrimitiveType::F32, {0}), 0, 1)
                        .value(),
                    bounded_shape);
}

XLA_TEST_F(CopyOpTest, CopyDynamicR1S1310720U32Dynamic106632) {
  // TODO(vsytch): CPU emitter doesn't handle dynamic shapes.
  if (platform()->Name() == "Host") {
    GTEST_SKIP();
  }
  Shape bounded_shape =
      ShapeUtil::MakeShape(PrimitiveType::F32, {1310720}, {true});
  TestDynamicCopyOp(
      LiteralUtil::CreateRandomLiteral<PrimitiveType::F32>(
          ShapeUtil::MakeShape(PrimitiveType::F32, {106632}), 0, 1)
          .value(),
      bounded_shape);
}

XLA_TEST_F(CopyOpTest, CopyDynamicR1S1310720U32Dynamic1310720) {
  // TODO(vsytch): CPU emitter doesn't handle dynamic shapes.
  if (platform()->Name() == "Host") {
    GTEST_SKIP();
  }
  Shape bounded_shape =
      ShapeUtil::MakeShape(PrimitiveType::F32, {1310720}, {true});
  TestDynamicCopyOp(
      LiteralUtil::CreateRandomLiteral<PrimitiveType::F32>(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1310720}), 0, 1)
          .value(),
      bounded_shape);
}

XLA_TEST_F(CopyOpTest, CopyDynamicR1S512U32Dynamic64) {
  // TODO(vsytch): CPU emitter doesn't handle dynamic shapes.
  if (platform()->Name() == "Host") {
    GTEST_SKIP();
  }
  Shape bounded_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {512}, {true});
  TestDynamicCopyOp(LiteralUtil::CreateRandomLiteral<PrimitiveType::F32>(
                        ShapeUtil::MakeShape(PrimitiveType::F32, {64}), 0, 1)
                        .value(),
                    bounded_shape);
}

XLA_TEST_F(CopyOpTest, CopyR3F32_2x2x3) {
  TestCopyOp(LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                                    {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

XLA_TEST_F(CopyOpTest, CopyR4S32_2x2x3x2) {
  TestCopyOp(LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

XLA_TEST_F(CopyOpTest, CopyR4S32_0x2x3x2) {
  TestCopyOp(LiteralUtil::CreateR4FromArray4D(Array4D<int32_t>(0, 2, 3, 2)));
}

XLA_TEST_F(CopyOpTest, CopyParameterScalar) {
  auto builder = HloComputation::Builder(TestName());

  // Copy literal to device to use as parameter.
  auto literal = LiteralUtil::CreateR0<float>(42.0);
  Shape shape = literal.shape();

  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, param0));

  auto computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));

  Literal result = ExecuteAndTransfer(std::move(module), {&literal});
  LiteralTestUtil::ExpectR0Near<float>(42.0f, result, ErrorSpec{0.0001});
}

XLA_TEST_F(CopyOpTest, CopyConstantR2Twice) {
  auto builder = HloComputation::Builder(TestName());

  auto literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  auto copy = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));
  builder.AddInstruction(
      HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, copy));

  auto computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  Literal result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 2.0}, {3.0, 4.0}}, result,
                                       ErrorSpec{0.0001});
}

XLA_TEST_F(CopyOpTest, CopyConstantR2DifferentLayouts) {
  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  // Reverse the minor-to-major order of the literal.
  Layout* literal_layout = literal.mutable_shape_do_not_use()->mutable_layout();
  ASSERT_EQ(2, literal_layout->minor_to_major_size());
  // Swap the first and second elements.
  *literal_layout->mutable_minor_to_major() = {
      literal_layout->minor_to_major(1), literal_layout->minor_to_major(0)};

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  Literal result = ExecuteAndTransfer(std::move(module), {});

  // The result of the computation has the default layout, which is the inverse
  // of the layout of the source literal.
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 3.0}, {2.0, 4.0}}, result,
                                       ErrorSpec{0.0001});
}

void CopyOpTest::TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3) {
  Array3D<int32_t> a(n1, n2, n3);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        a(i, j, k) = i * n3 * n2 + j * n3 + k;
      }
    }
  }

  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR3FromArray3D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({1, 2, 0}));
  Literal result = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectR3EqualArray3D(a, result);
}

void CopyOpTest::TestCopyConstantLayoutR4(
    size_t n1, size_t n2, size_t n3, size_t n4,
    absl::Span<const int64_t> permutation) {
  Array4D<int32_t> a(n1, n2, n3, n4);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        for (size_t l = 0; l < n4; ++l) {
          a(i, j, k, l) = i * n4 * n3 * n2 + j * n4 * n3 + k * n4 + l;
        }
      }
    }
  }

  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR4FromArray4D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout(permutation));
  Literal result = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectR4EqualArray4D(a, result);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_SingleIncompleteTilePerLayer) {
  TestCopyConstantLayout021(2, 2, 3);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_SingleCompleteTilePerLayer) {
  TestCopyConstantLayout021(2, 32, 32);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_MultipleTilesPerLayer) {
  TestCopyConstantLayout021(2, 70, 35);
}

XLA_TEST_F(CopyOpTest, CopyConstantR4Layout0231_MultipleTilesPerLayer) {
  TestCopyConstantLayoutR4(2, 70, 7, 5, {0, 2, 3, 1});
}

XLA_TEST_F(CopyOpTest, CopyConstantR4Layout0312_MultipleTilesPerLayer) {
  TestCopyConstantLayoutR4(2, 14, 5, 35, {0, 3, 1, 2});
}

using CopyOpClientTest = ClientLibraryTestBase;

XLA_TEST_F(CopyOpClientTest, Copy0x0) {
  Shape in_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {0, 0}, {0, 1});
  Shape out_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {0, 0}, {1, 0});
  auto empty = Literal::CreateFromShape(in_shape);

  XlaBuilder builder(TestName());
  Parameter(&builder, 0, in_shape, "input");
  auto input_data = client_->TransferToServer(empty).value();

  auto actual =
      ExecuteAndTransfer(&builder, {input_data.get()}, &out_shape).value();
  EXPECT_TRUE(LiteralTestUtil::Equal(empty, actual));
}

}  // namespace
}  // namespace xla
