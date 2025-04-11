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

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

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

}  // namespace
}  // namespace xla
