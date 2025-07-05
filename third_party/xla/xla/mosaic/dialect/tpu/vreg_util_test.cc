/* Copyright 2024 The JAX Authors.

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

#include "xla/mosaic/dialect/tpu/vreg_util.h"

#include <array>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;

MATCHER_P2(IsConstantOpWithSplatOrScalarValue, type, value, "") {
  auto constant_op = dyn_cast<arith::ConstantOp>(arg.getDefiningOp());
  if (constant_op == nullptr) {
    *result_listener << "Expected a constant op, got " << debugString(arg);
    return false;
  }

  return llvm::TypeSwitch<Attribute, bool>(constant_op.getValue())
      .template Case<DenseElementsAttr>([&](auto attr) {
        // If it's dense, it must be splat.
        if (attr.getType() != type) {
          *result_listener << "Expected a dense elements attr with type "
                           << debugString(type) << ", got "
                           << debugString(attr.getType());
          return false;
        }
        if (!attr.isSplat()) {
          *result_listener << "Expected a splat dense elements attr, got "
                           << debugString(attr);
          return false;
        }
        if (auto s = attr.template getSplatValue<decltype(value)>();
            s != value) {
          *result_listener << "Expected a splat dense elements attr with value "
                           << value << ", got " << s;
          return false;
        }
        return true;
      })
      .template Case<IntegerAttr>([&](auto attr) {
        if (attr.getType() != type) {
          *result_listener << "Expected a attr with type " << debugString(type)
                           << ", got " << debugString(attr.getType());
          return false;
        }
        if (auto s = attr.getInt(); s != value) {
          *result_listener << "Expected a attr with value " << value << ", got "
                           << s;
          return false;
        }
        return true;
      })
      .Default([&](auto attr) {
        *result_listener << "Unsupported attribute type: " << debugString(attr);
        return false;
      });
}

MATCHER_P2(IsVectorTypeWithShape, shape, elem_ty, "") {
  auto vty = dyn_cast<VectorType>(arg);
  if (vty == nullptr) {
    *result_listener << "Expected a vector type, got " << debugString(arg);
    return false;
  }
  if (vty.getShape() != ArrayRef<int64_t>(shape)) {
    *result_listener << "Expected a vector type with shape "
                     << absl::StrJoin(shape, ",") << ", got "
                     << absl::StrJoin(vty.getShape(), ",");
    return false;
  }
  if (vty.getElementType() != elem_ty) {
    *result_listener << "Expected a vector type with element type "
                     << debugString(elem_ty) << ", got "
                     << debugString(vty.getElementType());
    return false;
  }
  return true;
}

class VregUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<arith::ArithDialect, vector::VectorDialect,
                         tpu::TPUDialect>();
    mlir::Location loc = mlir::UnknownLoc::get(&context_);
    mlir::OpBuilder b(&context_);
    module_ = b.create<ModuleOp>(loc);
    builder_ = std::make_unique<mlir::ImplicitLocOpBuilder>(
        module_->getLoc(), module_->getBodyRegion());
  }

  void TearDown() override {
    builder_.reset();
    // Reset the module to prevent memory leaks.
    module_ = nullptr;
  }

  mlir::ImplicitLocOpBuilder& Builder() { return *builder_; }

 private:
  MLIRContext context_;
  std::unique_ptr<mlir::ImplicitLocOpBuilder> builder_;
  OwningOpRef<ModuleOp> module_;
};

TEST_F(VregUtilTest, GetNativeVregOrVmaskTypeBitwidthMismatch) {
  EXPECT_DEATH(getNativeVregOrVmaskType(Builder().getI16Type(),
                                        /*layout_bitwidth=*/8, {2, 4}),
               "");
}

TEST_F(VregUtilTest, GetNativeVregOrVmaskTypeI1) {
  EXPECT_THAT(getNativeVregOrVmaskType(Builder().getI1Type(),
                                       /*layout_bitwidth=*/8, {2, 4}),
              IsVectorTypeWithShape(std::array<int64_t, 3>{2, 4, 4},
                                    Builder().getI1Type()));
}

TEST_F(VregUtilTest, GetNativeVregF32) {
  EXPECT_THAT(getNativeVregType(Builder().getF32Type(), {2, 4}),
              IsVectorTypeWithShape(std::array<int64_t, 2>{2, 4},
                                    Builder().getF32Type()));
}

TEST_F(VregUtilTest, GetNativeVregBf16) {
  EXPECT_THAT(getNativeVregType(Builder().getBF16Type(), {2, 4}),
              IsVectorTypeWithShape(std::array<int64_t, 3>{2, 4, 2},
                                    Builder().getBF16Type()));
}

TEST_F(VregUtilTest, GetFullVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getI32Type());
  TypedValue<VectorType> vec =
      getFullVector(Builder(), vty, Builder().getI32IntegerAttr(0x1));

  EXPECT_THAT(vec, IsConstantOpWithSplatOrScalarValue(vty, int32_t{0x1}));
}

TEST_F(VregUtilTest, GetFullLikeVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getF32Type());
  TypedValue<VectorType> in_vec = Builder().create<vector::SplatOp>(
      vty, Builder().create<arith::ConstantOp>(
               vty.getElementType(), Builder().getF32FloatAttr(1.0f)));
  TypedValue<VectorType> vec =
      getFullLikeVector(Builder(), in_vec, Builder().getF32FloatAttr(2.0f));

  EXPECT_THAT(vec, IsConstantOpWithSplatOrScalarValue(vty, float{2.0f}));
}

TEST_F(VregUtilTest, GetZerosVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getI32Type());
  TypedValue<VectorType> vec = getZerosVector(Builder(), vty);

  EXPECT_THAT(vec, IsConstantOpWithSplatOrScalarValue(vty, int32_t{0}));
}

TEST_F(VregUtilTest, GetZerosLikeVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getF32Type());
  TypedValue<VectorType> in_vec = Builder().create<vector::SplatOp>(
      vty, Builder().create<arith::ConstantOp>(
               vty.getElementType(), Builder().getF32FloatAttr(1.0f)));
  TypedValue<VectorType> vec = getZerosLikeVector(Builder(), in_vec);

  EXPECT_THAT(vec, IsConstantOpWithSplatOrScalarValue(vty, float{0.0f}));
}

TEST_F(VregUtilTest, GetX32VmaskByPaddingEndDim0) {
  constexpr std::array<int64_t, 2> kTargetShape = {4, 8};
  FailureOr<TypedValue<VectorType>> vec = getX32VmaskByPaddingEnd(
      Builder(), /*padding=*/1, /*target_shape=*/kTargetShape,
      /*dim=*/0);
  ASSERT_TRUE(succeeded(vec));

  auto mask_op = dyn_cast<tpu::CreateMaskOp>(vec.value().getDefiningOp());
  ASSERT_TRUE(mask_op != nullptr);
  EXPECT_THAT(ArrayRef<Value>({mask_op.getLow()[0], mask_op.getLow()[1]}),
              ElementsAre(IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{0}),
                          IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{0})));
  EXPECT_THAT(ArrayRef<Value>({mask_op.getHigh()[0], mask_op.getHigh()[1]}),
              ElementsAre(IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{3}),
                          IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{8})));
}

TEST_F(VregUtilTest, GetX32VmaskByPaddingEndDim1) {
  constexpr std::array<int64_t, 2> kTargetShape = {4, 8};
  FailureOr<TypedValue<VectorType>> vec = getX32VmaskByPaddingEnd(
      Builder(), /*padding=*/3, /*target_shape=*/kTargetShape,
      /*dim=*/1);
  ASSERT_TRUE(succeeded(vec));

  auto mask_op = dyn_cast<tpu::CreateMaskOp>(vec.value().getDefiningOp());
  ASSERT_TRUE(mask_op != nullptr);
  EXPECT_THAT(ArrayRef<Value>({mask_op.getLow()[0], mask_op.getLow()[1]}),
              ElementsAre(IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{0}),
                          IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{0})));
  EXPECT_THAT(ArrayRef<Value>({mask_op.getHigh()[0], mask_op.getHigh()[1]}),
              ElementsAre(IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{4}),
                          IsConstantOpWithSplatOrScalarValue(
                              Builder().getIndexType(), int64_t{5})));
}

}  // namespace

}  // namespace mlir::tpu
