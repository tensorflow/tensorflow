/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"

#include <memory>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {
namespace {

template <int NInput, int NOutput>
FuncOp createMaxUnpoolingFunc(
    mlir::Builder* builder, const SmallVector<mlir::Type, NInput>& input_types,
    const SmallVector<mlir::Type, NOutput>& output_types) {
  auto func_type = builder->getFunctionType(input_types, output_types);
  auto func =
      FuncOp::create(mlir::NameLoc::get(builder->getIdentifier("fused_func")),
                     "fused_func", func_type, {});

  func.addEntryBlock();
  mlir::StringAttr attr_value = builder->getStringAttr("MaxUnpooling2D");
  func->setAttr("tf._implements", attr_value);
  return func;
}

FuncOp createMaxUnpoolingFunc(mlir::Builder* builder,
                              const SmallVector<int64_t, 4>& input_shape,
                              const SmallVector<int64_t, 4>& output_shape) {
  auto input_type = RankedTensorType::get(input_shape, builder->getF32Type());
  auto indices_type = RankedTensorType::get(input_shape, builder->getI64Type());
  auto output_type = RankedTensorType::get(output_shape, builder->getF32Type());
  SmallVector<mlir::Type, 2> input_types{input_type, indices_type};
  SmallVector<mlir::Type, 1> output_types{output_type};
  return createMaxUnpoolingFunc<2, 1>(builder, input_types, output_types);
}

template <int N>
ArrayAttr createInt32Array(mlir::Builder* builder, mlir::MLIRContext* context,
                           const SmallVector<int32_t, N>& values) {
  SmallVector<Attribute, N> ret;
  for (int32_t value : values) {
    ret.push_back(builder->getI32IntegerAttr(value));
  }
  return ArrayAttr::get(context, ret);
}

template <int N>
ArrayAttr createInt64Array(mlir::Builder* builder, mlir::MLIRContext* context,
                           const SmallVector<int64_t, N>& values) {
  SmallVector<Attribute, N> ret;
  for (int64_t value : values) {
    ret.push_back(builder->getI64IntegerAttr(value));
  }
  return ArrayAttr::get(context, ret);
}

mlir::TF::FuncAttr createMaxUnpoolingAttr(mlir::MLIRContext* context,
                                          const std::string& padding,
                                          const ArrayAttr& pool_size,
                                          const ArrayAttr& strides) {
  SmallVector<::mlir::NamedAttribute, 3> fields;

  auto padding_id = ::mlir::Identifier::get("padding", context);
  fields.emplace_back(padding_id, StringAttr::get(context, padding));

  auto pool_size_id = ::mlir::Identifier::get("pool_size", context);
  fields.emplace_back(pool_size_id, pool_size);

  auto strides_id = ::mlir::Identifier::get("strides", context);
  fields.emplace_back(strides_id, strides);

  DictionaryAttr dict = DictionaryAttr::get(context, fields);
  return TF::FuncAttr::get(context, "MaxUnpooling2D", dict);
}

}  // namespace

class PerceptionUtilsTest : public ::testing::Test {
 protected:
  PerceptionUtilsTest() {}

  void SetUp() override {
    context_ = std::make_unique<mlir::MLIRContext>();
    context_->loadDialect<mlir::StandardOpsDialect, mlir::TF::TensorFlowDialect,
                          TensorFlowLiteDialect>();
    builder_ = std::unique_ptr<mlir::Builder>(new Builder(context_.get()));

    fused_max_unpooling_func_ =
        createMaxUnpoolingFunc(builder_.get(), {2, 4, 4, 2}, {2, 2, 2, 2});

    func_attr_ = createMaxUnpoolingAttr(
        context_.get(), "SAME",
        createInt32Array<2>(builder_.get(), context_.get(), {2, 2}),
        createInt32Array<2>(builder_.get(), context_.get(), {2, 2}));
  }

  void TearDown() override {
    fused_max_unpooling_func_.erase();
    builder_.reset();
  }

  FuncOp fused_max_unpooling_func_;
  mlir::TF::FuncAttr func_attr_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::Builder> builder_;
};

TEST_F(PerceptionUtilsTest, VerifySignatureValid) {
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr_);

  EXPECT_FALSE(failed(convert.VerifySignature()));
}

TEST_F(PerceptionUtilsTest, VerifySignatureInvalid) {
  auto input_type = RankedTensorType::get({1, 2, 2, 1}, builder_->getF32Type());
  auto output_type =
      RankedTensorType::get({1, 2, 1, 1}, builder_->getF32Type());
  SmallVector<mlir::Type, 1> input_types{input_type};
  SmallVector<mlir::Type, 1> output_types{output_type};

  auto max_unpooling_func =
      createMaxUnpoolingFunc<1, 1>(builder_.get(), input_types, output_types);
  mlir::TFL::ConvertMaxUnpoolingFunc convert(max_unpooling_func, func_attr_);

  EXPECT_TRUE(failed(convert.VerifySignature()));
  max_unpooling_func->erase();
}

TEST_F(PerceptionUtilsTest, RewriteValid) {
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr_);

  EXPECT_FALSE(failed(convert.RewriteFunc()));
}

TEST_F(PerceptionUtilsTest, RewriteWrongPadding) {
  auto func_attr = createMaxUnpoolingAttr(
      context_.get(), "INVALID",
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}),
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}));
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr);

  EXPECT_TRUE(failed(convert.RewriteFunc()));
}

TEST_F(PerceptionUtilsTest, RewriteWrongFilter) {
  auto func_attr = createMaxUnpoolingAttr(
      context_.get(), "VALID",
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2, 2}),
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}));
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr);

  EXPECT_TRUE(failed(convert.RewriteFunc()));
}

TEST_F(PerceptionUtilsTest, RewriteWrongStrides) {
  auto func_attr = createMaxUnpoolingAttr(
      context_.get(), "VALID",
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}),
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2, 0}));
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr);

  EXPECT_TRUE(failed(convert.RewriteFunc()));
}

}  // namespace TFL
}  // namespace mlir
