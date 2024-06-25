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

#include "xla/translate/mhlo_to_hlo/type_to_shape.h"

#include <iostream>
#include <utility>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

using mlir::Builder;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::RankedTensorType;
using mlir::UnrankedTensorType;
using mlir::VectorType;

namespace xla {
namespace {

// Simple implementation of a proto matcher comparing string representations.
// Only works as ShapeProto's textual representation is deterministic.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tsl::protobuf::Message& expected)
      : expected_(expected.SerializeAsString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p, testing::MatchResultListener*) const {
    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tsl::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

TEST(TypeToShapeTest, ConvertBasicTypesToTypes) {
  MLIRContext context;
  Builder b(&context);

  EXPECT_TRUE(
      ShapeUtil::IsScalarWithElementType(TypeToShape(b.getF32Type()), F32));
  EXPECT_THAT(
      TypeToShape(VectorType::get({8, 128}, b.getIntegerType(32))).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::S32, {8, 128}).ToProto()));
  EXPECT_THAT(
      TypeToShape(VectorType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));

  // MLIR Type that is not representable as XLA Shape.
  EXPECT_THAT(
      TypeToShape(VectorType::get({8, 128}, b.getIntegerType(17))).ToProto(),
      EqualsProto(Shape().ToProto()));
}

TEST(TypeToShapeTest, ConvertMemRefTypeToTypes) {
  MLIRContext context;
  Builder b(&context);

  // Memref without any affine map. Note: memory space is ignored for shape.
  EXPECT_THAT(
      TypeToShape(MemRefType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));
  EXPECT_THAT(
      TypeToShape(MemRefType::get({100, 13, 210}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {100, 13, 210}).ToProto()));

  // Vector types are "flattened" into the end of the shape.
  EXPECT_THAT(
      TypeToShape(MemRefType::get({100, 13, 210},
                                  VectorType::get({8, 128}, b.getF32Type())))
          .ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {100, 13, 210, 8, 128})
              .ToProto()));
}

TEST(TypeToShapeTest, ConvertTensorTypeToTypes) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  Builder b(&context);

  EXPECT_THAT(
      TypeToShape(RankedTensorType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));

  llvm::SmallVector<int64_t, 4> bounds = {8, mlir::ShapedType::kDynamic};
  auto extensions = mlir::mhlo::TypeExtensionsAttr::get(&context, bounds);
  EXPECT_THAT(
      TypeToShape(RankedTensorType::get({mlir::ShapedType::kDynamic, 128},
                                        b.getF32Type(), extensions))
          .ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}, {true, false})
              .ToProto()));

  EXPECT_THAT(
      TypeToShape(RankedTensorType::get({mlir::ShapedType::kDynamic, 784},
                                        b.getF32Type()))
          .ToProto(),
      EqualsProto(ShapeUtil::MakeShape(PrimitiveType::F32,
                                       {Shape::kUnboundedSize, 784},
                                       {true, false})
                      .ToProto()));

  EXPECT_THAT(TypeToShape(UnrankedTensorType::get(b.getF32Type())).ToProto(),
              EqualsProto(Shape().ToProto()));

  // TODO(jpienaar): Expand to handle more complicated tensor types.
  EXPECT_THAT(
      TypeToShape(RankedTensorType::get(
                      {8, 128}, VectorType::get({16, 16}, b.getF32Type())))
          .ToProto(),
      EqualsProto(Shape().ToProto()));
}

TEST(TypeToShapeTest, ConvertMemRefToShape) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(PrimitiveType::F32,
                                                    {10, 20, 30}, {2, 0, 1});
  MLIRContext context;
  mlir::Builder builder(&context);

  absl::StatusOr<mlir::Type> mlir_type =
      ConvertShapeToType<MemRefType>(shape, builder);
  ASSERT_TRUE(mlir_type.ok());
  mlir::Type type = std::move(mlir_type).value();
  Shape converted = TypeToShape(type);
  EXPECT_TRUE(ShapeUtil::Equal(
      converted, ShapeUtil::MakeShapeWithDenseLayout(PrimitiveType::F32,
                                                     {10, 20, 30}, {2, 0, 1})));
  EXPECT_TRUE(ShapeUtil::Equal(converted, shape));
}

TEST(TypeToShapeTest, ConvertMemRefToShape2) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(PrimitiveType::C64,
                                                    {2, 4, 3, 3}, {2, 3, 1, 0});
  MLIRContext context;
  mlir::Builder builder(&context);

  absl::StatusOr<mlir::Type> mlir_type =
      ConvertShapeToType<MemRefType>(shape, builder);
  ASSERT_TRUE(mlir_type.ok());
  mlir::Type type = std::move(mlir_type).value();
  Shape converted = TypeToShape(type);
  EXPECT_TRUE(ShapeUtil::Equal(
      converted, ShapeUtil::MakeShapeWithDenseLayout(
                     PrimitiveType::C64, {2, 4, 3, 3}, {2, 3, 1, 0})));
  EXPECT_TRUE(ShapeUtil::Equal(converted, shape));
}

}  // namespace
}  // namespace xla
