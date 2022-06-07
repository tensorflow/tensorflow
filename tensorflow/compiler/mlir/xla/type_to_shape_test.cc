/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/type_to_shape.h"

#include <iostream>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"

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
  explicit ProtoStringMatcher(const tensorflow::protobuf::Message& expected)
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
    const tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

TEST(TypeToShapeTest, ConvertPrimitiveTypes) {
  MLIRContext context;
  Builder b(&context);

  EXPECT_EQ(TypeToPrimitiveType(b.getF32Type()), PrimitiveType::F32);
  EXPECT_EQ(TypeToPrimitiveType(b.getIntegerType(1)), PrimitiveType::PRED);
  EXPECT_EQ(TypeToPrimitiveType(b.getIntegerType(17)),
            PrimitiveType::PRIMITIVE_TYPE_INVALID);
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
  MLIRContext context;
  Builder b(&context);

  EXPECT_THAT(
      TypeToShape(RankedTensorType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));

  // Shape cannot represent dynamic shapes.
  // TODO(b/115638799): Update once Shape can support dynamic shapes.
  EXPECT_THAT(TypeToShape(UnrankedTensorType::get(b.getF32Type())).ToProto(),
              EqualsProto(Shape().ToProto()));

  // TODO(jpienaar): Expand to handle more complicated tensor types.
  EXPECT_THAT(
      TypeToShape(RankedTensorType::get(
                      {8, 128}, VectorType::get({16, 16}, b.getF32Type())))
          .ToProto(),
      EqualsProto(Shape().ToProto()));
}

TEST(TypeToShapeTest, ConvertWithShapeRepresentationFn) {
  tensorflow::DataType captured_dtype;
  tensorflow::TensorShape captured_tensor_shape;

  // A dummy shape representation function that does nothing other than
  // capturing arguments passed to it.
  auto test_shape_representation_fn = [&](const tensorflow::TensorShape& shape,
                                          tensorflow::DataType dtype) {
    captured_tensor_shape = shape;
    captured_dtype = dtype;
    return xla::Shape();
  };

  MLIRContext context;
  Builder b(&context);
  StatusOr<Shape> status_or_shape;

  // Non-fully-defined shape.
  status_or_shape =
      TypeToShape(RankedTensorType::get({-1, 2, 3}, b.getF32Type()),
                  test_shape_representation_fn);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(status_or_shape.status()));

  // Scalar Int32 Tensor, using fast memory.
  status_or_shape =
      TypeToShape(b.getIntegerType(32), test_shape_representation_fn);
  EXPECT_TRUE(status_or_shape.ok());
  EXPECT_EQ(captured_dtype, tensorflow::DataType::DT_INT32);
  EXPECT_EQ(captured_tensor_shape, tensorflow::TensorShape());

  // Ranked Float32 Tensor, not using fast memory.
  status_or_shape =
      TypeToShape(RankedTensorType::get({1, 2, 3}, b.getF32Type()),
                  test_shape_representation_fn);
  EXPECT_TRUE(status_or_shape.ok());
  EXPECT_EQ(captured_dtype, tensorflow::DataType::DT_FLOAT);
  EXPECT_EQ(captured_tensor_shape, tensorflow::TensorShape({1, 2, 3}));
}

TEST(TypeToShapeTest, ConvertMemRefToShape) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(PrimitiveType::F32, {10, 20, 30},
                                               {2, 0, 1});
  MLIRContext context;
  mlir::Builder builder(&context);

  StatusOr<mlir::Type> mlir_type =
      ConvertShapeToType<MemRefType>(shape, builder);
  ASSERT_TRUE(mlir_type.ok());
  mlir::Type type = mlir_type.ConsumeValueOrDie();
  Shape converted = TypeToShape(type);
  EXPECT_TRUE(ShapeUtil::Equal(
      converted, ShapeUtil::MakeShapeWithLayout(PrimitiveType::F32,
                                                {10, 20, 30}, {2, 0, 1})));
  EXPECT_TRUE(ShapeUtil::Equal(converted, shape));
}

TEST(TypeToShapeTest, ConvertMemRefToShape2) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(PrimitiveType::C64, {2, 4, 3, 3},
                                               {2, 3, 1, 0});
  MLIRContext context;
  mlir::Builder builder(&context);

  StatusOr<mlir::Type> mlir_type =
      ConvertShapeToType<MemRefType>(shape, builder);
  ASSERT_TRUE(mlir_type.ok());
  mlir::Type type = mlir_type.ConsumeValueOrDie();
  Shape converted = TypeToShape(type);
  EXPECT_TRUE(ShapeUtil::Equal(
      converted, ShapeUtil::MakeShapeWithLayout(PrimitiveType::C64,
                                                {2, 4, 3, 3}, {2, 3, 1, 0})));
  EXPECT_TRUE(ShapeUtil::Equal(converted, shape));
}

}  // namespace
}  // namespace xla
