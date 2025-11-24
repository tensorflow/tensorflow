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

#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"

#include <cstdint>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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

TEST(TypeToShapeTest, ConvertTensorTypeToTypes) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
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

  auto extensions_stablehlo =
      mlir::stablehlo::TypeExtensionsAttr::get(&context, bounds);
  EXPECT_THAT(
      TypeToShape(RankedTensorType::get({mlir::ShapedType::kDynamic, 128},
                                        b.getF32Type(), extensions_stablehlo))
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

TEST(TypeToShapeTest, ConvertBufferTypeToTypes) {
  MLIRContext context;
  Builder builder(&context);

  Shape shape1 = ShapeUtil::MakeShapeWithDenseLayout(PrimitiveType::F32,
                                                     {10, 20, 30}, {1, 0, 2});

  EXPECT_THAT(
      TypeToShape(ConvertShapeToType<MemRefType>(shape1, builder).value())
          .ToProto(),
      EqualsProto(
          ShapeUtil::MakeValidatedBufferShape(shape1).value().ToProto()));

  Shape shape2 = ShapeUtil::MakeShapeWithDenseLayout(PrimitiveType::F32,
                                                     {10, 20, 30}, {2, 1, 0});

  EXPECT_THAT(
      TypeToShape(ConvertShapeToType<MemRefType>(shape2, builder).value())
          .ToProto(),
      EqualsProto(
          ShapeUtil::MakeValidatedBufferShape(shape2).value().ToProto()));
}

}  // namespace
}  // namespace xla
