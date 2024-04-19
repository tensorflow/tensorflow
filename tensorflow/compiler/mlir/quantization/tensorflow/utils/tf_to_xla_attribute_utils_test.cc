/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_xla_attribute_utils.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir::quant {
namespace {

void PackOperandTestHelper(
    const llvm::SmallVector<int64_t>& unpacked_shape,
    const llvm::SmallVector<int8_t>& unpacked_values, int pack_dim,
    const llvm::SmallVector<int64_t>& expected_packed_shape,
    const llvm::SmallVector<int8_t>& expected_packed_values) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  OpBuilder builder(&module->getBodyRegion());
  context.loadDialect<TF::TensorFlowDialect>();

  Value value = CreateConstValue<int8_t>(builder, module->getLoc(),
                                         unpacked_shape, unpacked_values);
  Value packed_value = PackOperand(builder, module->getLoc(), value, pack_dim);
  DenseIntElementsAttr packed_value_attr;
  ASSERT_TRUE(matchPattern(packed_value, m_Constant(&packed_value_attr)));

  ShapedType packed_shape_type = packed_value.getType().dyn_cast<ShapedType>();
  llvm::SmallVector<int64_t> packed_shape(packed_shape_type.getShape().begin(),
                                          packed_shape_type.getShape().end());
  EXPECT_THAT(packed_shape, testing::ElementsAreArray(expected_packed_shape));
  llvm::SmallVector<int8_t> packed_value_vector(
      packed_value_attr.getValues<int8_t>());
  EXPECT_THAT(packed_value_vector,
              testing::ElementsAreArray(expected_packed_values));
}

TEST(TfToXlaAttributeUtilsTest, PackOperandPackDimSizeEven) {
  PackOperandTestHelper(/*unpacked_shape=*/{2, 2},
                        /*unpacked_values=*/{0x01, 0x02, 0x03, 0x04},
                        /*pack_dim=*/0,
                        /*expected_packed_shape=*/{1, 2},
                        /*expected_packed_values=*/{0x31, 0x42});
}

TEST(TfToXlaAttributeUtilsTest, PackOperandPackDimSizeOdd) {
  PackOperandTestHelper(
      /*unpacked_shape=*/{2, 3},
      /*unpacked_values=*/{0x01, 0x02, 0x03, 0x04, 0x05, 0x06},
      /*pack_dim=*/1,
      /*expected_packed_shape=*/{2, 2},
      /*expected_packed_values=*/{0x31, 0x02, 0x64, 0x05});
}

}  // namespace
}  // namespace mlir::quant
