/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class CopyOpTest : public HloTestBase {
 protected:
  void TestCopyOp(const Literal& literal) {
    auto builder = HloComputation::Builder(TestName());
    auto constant = builder.AddInstruction(
        HloInstruction::CreateConstant(literal.CloneToUnique()));
    builder.AddInstruction(HloInstruction::CreateUnary(
        constant->shape(), HloOpcode::kCopy, constant));
    auto computation = builder.Build();
    auto module = CreateNewModule();
    module->AddEntryComputation(std::move(computation));

    std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});
    EXPECT_TRUE(LiteralTestUtil::Equal(literal, *result));
  }

  void TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3);
  void TestCopyConstantLayoutR4(size_t n1, size_t n2, size_t n3, size_t n4,
                                tensorflow::gtl::ArraySlice<int64> permutation);
};

XLA_TEST_F(CopyOpTest, CopyR0Bool) {
  TestCopyOp(*LiteralUtil::CreateR0<bool>(true));
}

XLA_TEST_F(CopyOpTest, CopyR1S0U32) {
  TestCopyOp(*LiteralUtil::CreateR1<uint32>({}));
}

XLA_TEST_F(CopyOpTest, CopyR1S3U32) {
  TestCopyOp(*LiteralUtil::CreateR1<uint32>({1, 2, 3}));
}

XLA_TEST_F(CopyOpTest, CopyR3F32_2x2x3) {
  TestCopyOp(
      *LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                              {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

XLA_TEST_F(CopyOpTest, CopyR4S32_2x2x3x2) {
  TestCopyOp(*LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

XLA_TEST_F(CopyOpTest, CopyR4S32_0x2x3x2) {
  TestCopyOp(*LiteralUtil::CreateR4FromArray4D(Array4D<int32>(0, 2, 3, 2)));
}

XLA_TEST_F(CopyOpTest, CopyParameterScalar) {
  auto builder = HloComputation::Builder(TestName());

  // Copy literal to device to use as parameter.
  auto literal = LiteralUtil::CreateR0<float>(42.0);
  Shape shape = literal->shape();

  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, param0));

  auto computation = builder.Build();

  auto module = CreateNewModule();
  module->AddEntryComputation(std::move(computation));

  std::unique_ptr<Literal> result =
      ExecuteAndTransfer(std::move(module), {literal.get()});
  LiteralTestUtil::ExpectR0Near<float>(42.0f, *result, error_spec_);
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

  auto module = CreateNewModule();
  module->AddEntryComputation(std::move(computation));
  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 2.0}, {3.0, 4.0}}, *result,
                                       error_spec_);
}

XLA_TEST_F(CopyOpTest, CopyConstantR2DifferentLayouts) {
  HloComputation::Builder builder(TestName());

  std::unique_ptr<Literal> literal =
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  // Reverse the minor-to-major order of the literal.
  Layout* literal_layout =
      literal->mutable_shape_do_not_use()->mutable_layout();
  ASSERT_EQ(2, literal_layout->minor_to_major_size());
  literal_layout->mutable_minor_to_major()->SwapElements(0, 1);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewModule();
  module->AddEntryComputation(std::move(computation));
  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});

  // The result of the computation has the default layout, which is the inverse
  // of the layout of the source literal.
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 3.0}, {2.0, 4.0}}, *result,
                                       error_spec_);
}

void CopyOpTest::TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3) {
  Array3D<int32> a(n1, n2, n3);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        a(i, j, k) = i * n3 * n2 + j * n3 + k;
      }
    }
  }

  HloComputation::Builder builder(TestName());

  std::unique_ptr<Literal> literal = LiteralUtil::CreateR3FromArray3D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewModule();
  module->AddEntryComputation(std::move(computation));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({1, 2, 0}));
  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectR3EqualArray3D(a, *result);
}

void CopyOpTest::TestCopyConstantLayoutR4(
    size_t n1, size_t n2, size_t n3, size_t n4,
    tensorflow::gtl::ArraySlice<int64> permutation) {
  Array4D<int32> a(n1, n2, n3, n4);
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

  std::unique_ptr<Literal> literal = LiteralUtil::CreateR4FromArray4D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewModule();
  module->AddEntryComputation(std::move(computation));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout(permutation));
  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectR4EqualArray4D(a, *result);
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
  Shape in_shape = ShapeUtil::MakeShapeWithLayout(F32, {0, 0}, {0, 1});
  Shape out_shape = ShapeUtil::MakeShapeWithLayout(F32, {0, 0}, {1, 0});
  auto empty = Literal::CreateFromShape(in_shape);

  XlaBuilder builder(TestName());
  Parameter(&builder, 0, in_shape, "input");
  auto input_data = client_->TransferToServer(*empty).ConsumeValueOrDie();

  auto actual = ExecuteAndTransfer(&builder, {input_data.get()}, &out_shape)
                    .ConsumeValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Equal(*empty, *actual));
}

}  // namespace
}  // namespace xla
