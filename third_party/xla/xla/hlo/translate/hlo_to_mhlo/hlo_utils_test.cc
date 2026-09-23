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

#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

// Returns the storage bytes of `attr` as signed values.
std::vector<int8_t> RawBytes(mlir::DenseElementsAttr attr) {
  llvm::ArrayRef<char> raw = attr.getRawData();
  return std::vector<int8_t>(raw.begin(), raw.end());
}

// Returns the elements read the way MLIR consumers such as the printer read
// them, one APInt per element.
std::vector<int64_t> SubByteValues(mlir::DenseElementsAttr attr,
                                   bool is_signed) {
  std::vector<int64_t> values;
  for (const llvm::APInt& value : attr.getValues<llvm::APInt>()) {
    values.push_back(is_signed ? value.getSExtValue() : value.getZExtValue());
  }
  return values;
}

TEST(ConvertTensorShapeToType, Simple) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::Builder builder(&context);

  // Static shape.
  {
    auto shape = ShapeUtil::MakeShape(PrimitiveType::S32, {8, 128});
    TF_ASSERT_OK_AND_ASSIGN(
        auto type,
        ConvertTensorShapeToType<mlir::RankedTensorType>(shape, builder));

    auto expected = mlir::RankedTensorType::get({8, 128}, builder.getI32Type());
    EXPECT_TRUE(type == expected)
        << " Expected: " << mlir::debugString(expected)
        << " Computed: " << mlir::debugString(type);
  }

  // Dynamic shape.
  {
    auto shape =
        ShapeUtil::MakeShape(PrimitiveType::S32, {8, 128}, {true, false});
    TF_ASSERT_OK_AND_ASSIGN(
        auto type,
        ConvertTensorShapeToType<mlir::RankedTensorType>(shape, builder));

    int64_t bounds[] = {8, mlir::ShapedType::kDynamic};
    auto extensions =
        mlir::stablehlo::TypeExtensionsAttr::get(&context, bounds);
    auto expected = mlir::RankedTensorType::get(
        {mlir::ShapedType::kDynamic, 128}, builder.getI32Type(), extensions);
    EXPECT_TRUE(type == expected)
        << " Expected: " << mlir::debugString(expected)
        << " Computed: " << mlir::debugString(type);
  }
}

TEST(CreateDenseElementsAttrFromLiteral, SignedInt4IsSignExtendedPerByte) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  // Odd element count, both extremes and negative values.
  Literal literal = LiteralUtil::CreateR1<s4>(
      {s4(-8), s4(-1), s4(7), s4(0), s4(-3), s4(5), s4(-7)});

  ASSERT_OK_AND_ASSIGN(mlir::DenseElementsAttr attr,
                       CreateDenseElementsAttrFromLiteral(literal, builder));

  EXPECT_EQ(attr.getType(),
            mlir::RankedTensorType::get({7}, builder.getIntegerType(4)));
  EXPECT_FALSE(attr.isSplat());
  // One byte per element, holding the value sign extended to the full byte.
  EXPECT_THAT(RawBytes(attr), ElementsAre(-8, -1, 7, 0, -3, 5, -7));
  EXPECT_THAT(SubByteValues(attr, /*is_signed=*/true),
              ElementsAre(-8, -1, 7, 0, -3, 5, -7));
}

TEST(CreateDenseElementsAttrFromLiteral, UnsignedInt4IsZeroExtendedPerByte) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  Literal literal =
      LiteralUtil::CreateR1<u4>({u4(15), u4(0), u4(8), u4(7), u4(9)});

  ASSERT_OK_AND_ASSIGN(mlir::DenseElementsAttr attr,
                       CreateDenseElementsAttrFromLiteral(literal, builder));

  EXPECT_EQ(attr.getType(),
            mlir::RankedTensorType::get(
                {5}, builder.getIntegerType(4, /*isSigned=*/false)));
  EXPECT_THAT(RawBytes(attr), ElementsAre(15, 0, 8, 7, 9));
  EXPECT_THAT(SubByteValues(attr, /*is_signed=*/false),
              ElementsAre(15, 0, 8, 7, 9));
}

TEST(CreateDenseElementsAttrFromLiteral, SignedInt2) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  Literal literal = LiteralUtil::CreateR1<s2>({s2(-2), s2(-1), s2(1)});

  ASSERT_OK_AND_ASSIGN(mlir::DenseElementsAttr attr,
                       CreateDenseElementsAttrFromLiteral(literal, builder));

  EXPECT_EQ(attr.getType(),
            mlir::RankedTensorType::get({3}, builder.getIntegerType(2)));
  EXPECT_THAT(RawBytes(attr), ElementsAre(-2, -1, 1));
  EXPECT_THAT(SubByteValues(attr, /*is_signed=*/true), ElementsAre(-2, -1, 1));
}

TEST(CreateDenseElementsAttrFromLiteral, Int4WithNonDefaultLayoutIsRowMajor) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  // Column major storage; the attribute lists the elements in row major
  // (logical index) order.
  Literal literal = LiteralUtil::CreateR2WithLayout<s4>(
      {{s4(1), s4(-2), s4(3)}, {s4(-4), s4(5), s4(-6)}},
      LayoutUtil::MakeLayout({0, 1}));

  ASSERT_OK_AND_ASSIGN(mlir::DenseElementsAttr attr,
                       CreateDenseElementsAttrFromLiteral(literal, builder));

  EXPECT_EQ(attr.getType(),
            mlir::RankedTensorType::get({2, 3}, builder.getIntegerType(4)));
  EXPECT_THAT(RawBytes(attr), ElementsAre(1, -2, 3, -4, 5, -6));
}

TEST(StringRefToStringView, Conversion) {
  absl::string_view sv = "hello";
  llvm::StringRef sref = ToStringRef(sv);
  EXPECT_EQ(sref.data(), sv.data());
  EXPECT_EQ(sref.size(), sv.size());

  absl::string_view sv2 = ToStringView(sref);
  EXPECT_EQ(sv2.data(), sref.data());
  EXPECT_EQ(sv2.size(), sref.size());
}

}  // namespace
}  // namespace xla
