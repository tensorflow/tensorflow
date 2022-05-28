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

#include "tensorflow/compiler/mlir/xla/hlo_utils.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

TEST(ConvertTensorShapeToType, Simple) {
  mlir::MLIRContext context;
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
    auto shape = ShapeUtil::MakeShape(PrimitiveType::S32, {8, 128});
    shape.set_dynamic_dimension(0, true);

    TF_ASSERT_OK_AND_ASSIGN(
        auto type,
        ConvertTensorShapeToType<mlir::RankedTensorType>(shape, builder));
    auto expected = mlir::RankedTensorType::get(
        {mlir::ShapedType::kDynamicSize, 128}, builder.getI32Type());
    EXPECT_TRUE(type == expected)
        << " Expected: " << mlir::debugString(expected)
        << " Computed: " << mlir::debugString(type);
  }
}

}  // namespace
}  // namespace xla
