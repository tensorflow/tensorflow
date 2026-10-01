/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/translate/mhlo_to_hlo/literal_exporter.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

std::vector<llvm::APInt> SignedAPInts(unsigned bit_width,
                                      std::vector<int64_t> values) {
  std::vector<llvm::APInt> result;
  result.reserve(values.size());
  for (int64_t value : values) {
    result.push_back(llvm::APInt(bit_width, value, /*isSigned=*/true));
  }
  return result;
}

std::vector<llvm::APInt> UnsignedAPInts(unsigned bit_width,
                                        std::vector<uint64_t> values) {
  std::vector<llvm::APInt> result;
  result.reserve(values.size());
  for (uint64_t value : values) {
    result.push_back(llvm::APInt(bit_width, value));
  }
  return result;
}

// Attribute storage bytes as the HLO importer writes them: one sign extended
// byte per element.
std::vector<char> SignExtendedRawBuffer(std::vector<int8_t> values) {
  return std::vector<char>(values.begin(), values.end());
}

// The bytes a sub byte literal holds in memory, one per element.
std::vector<uint8_t> StorageBytes(const xla::Literal& literal) {
  const uint8_t* data = static_cast<const uint8_t*>(literal.untyped_data());
  return std::vector<uint8_t>(data, data + literal.size_bytes());
}

TEST(CreateLiteralFromAttribute, SignedInt4) {
  MLIRContext context;
  Builder builder(&context);
  // Odd element count, both extremes and negative values.
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({7}, builder.getIntegerType(4)),
      llvm::ArrayRef<llvm::APInt>(SignedAPInts(4, {-8, -1, 7, 0, -3, 5, -7})));

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(
      xla::LiteralUtil::CreateR1<xla::s4>({xla::s4(-8), xla::s4(-1), xla::s4(7),
                                           xla::s4(0), xla::s4(-3), xla::s4(5),
                                           xla::s4(-7)}),
      /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, SignedInt4IgnoresHighBitsOfStorageBytes) {
  MLIRContext context;
  Builder builder(&context);
  std::vector<char> raw = SignExtendedRawBuffer({-8, -1, 7, 0, -3, 5});
  DenseElementsAttr attr = DenseElementsAttr::getFromRawBuffer(
      RankedTensorType::get({2, 3}, builder.getIntegerType(4)), raw);

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(xla::LiteralUtil::CreateR2<xla::s4>(
                                {{xla::s4(-8), xla::s4(-1), xla::s4(7)},
                                 {xla::s4(0), xla::s4(-3), xla::s4(5)}}),
                            /*layout_sensitive=*/true));
  // Storage holds the low four bits of each value.
  EXPECT_THAT(StorageBytes(literal), ElementsAre(8, 15, 7, 0, 13, 5));
}

TEST(CreateLiteralFromAttribute, SplatInt4) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({2, 3}, builder.getIntegerType(4)),
      llvm::APInt(4, -3, /*isSigned=*/true));
  ASSERT_TRUE(attr.isSplat());

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(xla::LiteralUtil::CreateR2<xla::s4>(
                                {{xla::s4(-3), xla::s4(-3), xla::s4(-3)},
                                 {xla::s4(-3), xla::s4(-3), xla::s4(-3)}}),
                            /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, SplatInt4WithNonDefaultLayout) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({2, 3}, builder.getIntegerType(4)),
      llvm::APInt(4, -6, /*isSigned=*/true));
  ASSERT_TRUE(attr.isSplat());

  ASSERT_OK_AND_ASSIGN(
      xla::Literal literal,
      CreateLiteralFromAttribute(attr, xla::LayoutUtil::MakeLayout({0, 1})));

  EXPECT_EQ(literal.shape(),
            xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S4, {2, 3}, {0, 1}));
  EXPECT_TRUE(literal.Equal(xla::LiteralUtil::CreateR2WithLayout<xla::s4>(
                                {{xla::s4(-6), xla::s4(-6), xla::s4(-6)},
                                 {xla::s4(-6), xla::s4(-6), xla::s4(-6)}},
                                xla::LayoutUtil::MakeLayout({0, 1})),
                            /*layout_sensitive=*/true));
  // Storage holds the low four bits of the value in every slot.
  EXPECT_THAT(StorageBytes(literal), ElementsAre(10, 10, 10, 10, 10, 10));
}

TEST(CreateLiteralFromAttribute, SplatFloatWithNonDefaultLayout) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({2, 3}, builder.getF32Type()), 2.5f);
  ASSERT_TRUE(attr.isSplat());

  ASSERT_OK_AND_ASSIGN(
      xla::Literal literal,
      CreateLiteralFromAttribute(attr, xla::LayoutUtil::MakeLayout({0, 1})));

  EXPECT_EQ(literal.shape(),
            xla::ShapeUtil::MakeShapeWithDenseLayout(xla::F32, {2, 3}, {0, 1}));
  EXPECT_TRUE(literal.Equal(xla::LiteralUtil::CreateR2WithLayout<float>(
                                {{2.5f, 2.5f, 2.5f}, {2.5f, 2.5f, 2.5f}},
                                xla::LayoutUtil::MakeLayout({0, 1})),
                            /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, SplatFloat4E2M1FN) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({3}, builder.getType<Float4E2M1FNType>()),
      llvm::APFloat(llvm::APFloat::Float4E2M1FN(), "-1.5"));
  ASSERT_TRUE(attr.isSplat());

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(
      literal.Equal(xla::LiteralUtil::CreateR1<xla::float4_e2m1fn>(
                        {xla::float4_e2m1fn(-1.5f), xla::float4_e2m1fn(-1.5f),
                         xla::float4_e2m1fn(-1.5f)}),
                    /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, UnsignedInt4) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({5}, builder.getIntegerType(4, /*isSigned=*/false)),
      llvm::ArrayRef<llvm::APInt>(UnsignedAPInts(4, {15, 0, 8, 7, 9})));

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(
      xla::LiteralUtil::CreateR1<xla::u4>(
          {xla::u4(15), xla::u4(0), xla::u4(8), xla::u4(7), xla::u4(9)}),
      /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, SignedInt2) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({3}, builder.getIntegerType(2)),
      llvm::ArrayRef<llvm::APInt>(SignedAPInts(2, {-2, -1, 1})));

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(xla::LiteralUtil::CreateR1<xla::s2>(
                                {xla::s2(-2), xla::s2(-1), xla::s2(1)}),
                            /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, Int4WithNonDefaultLayout) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({2, 3}, builder.getIntegerType(4)),
      llvm::ArrayRef<llvm::APInt>(SignedAPInts(4, {1, -2, 3, -4, 5, -6})));

  ASSERT_OK_AND_ASSIGN(
      xla::Literal literal,
      CreateLiteralFromAttribute(attr, xla::LayoutUtil::MakeLayout({0, 1})));

  EXPECT_EQ(literal.shape(),
            xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S4, {2, 3}, {0, 1}));
  EXPECT_TRUE(literal.Equal(xla::LiteralUtil::CreateR2WithLayout<xla::s4>(
                                {{xla::s4(1), xla::s4(-2), xla::s4(3)},
                                 {xla::s4(-4), xla::s4(5), xla::s4(-6)}},
                                xla::LayoutUtil::MakeLayout({0, 1})),
                            /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, Float) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({2, 3}, builder.getF32Type()),
      llvm::ArrayRef<float>({1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}));
  xla::Literal expected = xla::LiteralUtil::CreateR2<float>(
      {{1.0f, -2.0f, 3.0f}, {-4.0f, 5.0f, -6.0f}});

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));
  EXPECT_TRUE(literal.Equal(expected, /*layout_sensitive=*/true));

  // An explicit descending layout gives the same literal.
  ASSERT_OK_AND_ASSIGN(
      xla::Literal descending,
      CreateLiteralFromAttribute(attr, xla::LayoutUtil::MakeLayout({1, 0})));
  EXPECT_TRUE(descending.Equal(expected, /*layout_sensitive=*/true));

  ASSERT_OK_AND_ASSIGN(
      xla::Literal column_major,
      CreateLiteralFromAttribute(attr, xla::LayoutUtil::MakeLayout({0, 1})));
  EXPECT_EQ(column_major.shape(),
            xla::ShapeUtil::MakeShapeWithDenseLayout(xla::F32, {2, 3}, {0, 1}));
  EXPECT_TRUE(
      column_major.Equal(xla::LiteralUtil::CreateR2WithLayout<float>(
                             {{1.0f, -2.0f, 3.0f}, {-4.0f, 5.0f, -6.0f}},
                             xla::LayoutUtil::MakeLayout({0, 1})),
                         /*layout_sensitive=*/true));
}

TEST(CreateLiteralFromAttribute, EmptyInt4) {
  MLIRContext context;
  Builder builder(&context);
  DenseElementsAttr attr = DenseElementsAttr::get(
      RankedTensorType::get({0, 4}, builder.getIntegerType(4)),
      llvm::ArrayRef<llvm::APInt>());

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_EQ(literal.shape(), xla::ShapeUtil::MakeShape(xla::S4, {0, 4}));
  EXPECT_EQ(literal.element_count(), 0);
}

// The HLO importer writes sign extended bytes; the exporter must read them
// back to the same literal, with the low bits only in storage.
TEST(CreateLiteralFromAttribute, RoundTripInt4) {
  MLIRContext context;
  Builder builder(&context);
  xla::Literal original = xla::LiteralUtil::CreateR2<xla::s4>(
      {{xla::s4(-8), xla::s4(-1), xla::s4(7)},
       {xla::s4(0), xla::s4(-3), xla::s4(5)}});
  ASSERT_OK_AND_ASSIGN(
      DenseElementsAttr attr,
      xla::CreateDenseElementsAttrFromLiteral(original, builder));
  ASSERT_FALSE(attr.isSplat());

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(original, /*layout_sensitive=*/true));
  EXPECT_THAT(StorageBytes(literal), ElementsAreArray(StorageBytes(original)));
}

// MLIR collapses a uniform raw buffer into a one element splat, so uniform
// constants reach the splat path of the exporter.
TEST(CreateLiteralFromAttribute, RoundTripUniformInt4) {
  MLIRContext context;
  Builder builder(&context);
  xla::Literal original = xla::LiteralUtil::CreateR2<xla::s4>(
      {{xla::s4(-6), xla::s4(-6), xla::s4(-6)},
       {xla::s4(-6), xla::s4(-6), xla::s4(-6)}});
  ASSERT_OK_AND_ASSIGN(
      DenseElementsAttr attr,
      xla::CreateDenseElementsAttrFromLiteral(original, builder));
  ASSERT_TRUE(attr.isSplat());

  ASSERT_OK_AND_ASSIGN(xla::Literal literal,
                       CreateLiteralFromAttribute(attr, {}));

  EXPECT_TRUE(literal.Equal(original, /*layout_sensitive=*/true));
  EXPECT_THAT(StorageBytes(literal), ElementsAreArray(StorageBytes(original)));
}

}  // namespace
}  // namespace mhlo
}  // namespace mlir
