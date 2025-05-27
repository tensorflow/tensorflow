/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/codegen/emitters/type_util.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace emitters {
namespace {

using ::testing::ElementsAre;

std::string TypeToString(mlir::Type type) {
  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << type;
  return out;
}

llvm::SmallVector<std::string> TypesToString(
    const llvm::SmallVector<mlir::Type>& types) {
  return llvm::map_to_vector(types, TypeToString);
}

TEST(TensorShapeTest, ConvertsShape) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder b(&ctx);
  EXPECT_EQ(TypeToString(
                TensorShapeToMlirType(ShapeUtil::MakeShape(S32, {4, 5, 6}), b)),
            "tensor<4x5x6xi32>");
}

TEST(TensorShapeTest, ConvertsPred) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder b(&ctx);
  EXPECT_EQ(TypeToString(TensorShapeToMlirType(
                ShapeUtil::MakeShape(PRED, {4, 5, 6}), b)),
            "tensor<4x5x6xi8>");
}

TEST(TensorShapeTest, ConvertsLayout) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder b(&ctx);
  EXPECT_EQ(
      TypeToString(TensorShapeToMlirType(
          ShapeUtil::MakeShapeWithDenseLayout(S32, {4, 5, 6}, {0, 2, 1}), b)),
      "tensor<4x5x6xi32, dense<[0, 2, 1]> : tensor<3xi64>>");
}

TEST(ShapeTest, ConvertsArray) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder b(&ctx);
  EXPECT_THAT(
      TypesToString(ShapeToMlirTypes(ShapeUtil::MakeShape(S32, {4, 5, 6}), b)),
      ElementsAre("tensor<4x5x6xi32>"));
}

TEST(ShapeTest, ConvertsTuple) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder b(&ctx);

  EXPECT_THAT(
      TypesToString(ShapeToMlirTypes(
          ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {4, 5, 6}),
                                     ShapeUtil::MakeShape(F32, {})}),
          b)),
      ElementsAre("tensor<4x5x6xi32>", "tensor<f32>"));
}

}  // namespace
}  // namespace emitters
}  // namespace xla
