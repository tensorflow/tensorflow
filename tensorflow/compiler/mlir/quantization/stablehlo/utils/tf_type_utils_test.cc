/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tsl/framework/numeric_types.h"

namespace mlir::quant::tensorflow {
namespace {

std::string GetQint8Tensor() {
  ::tensorflow::Tensor tensor(::tensorflow::DT_QINT8, {2, 2});
  tensor.matrix<tsl::qint8>()(0, 0) = tsl::qint8(1);
  tensor.matrix<tsl::qint8>()(0, 1) = tsl::qint8(2);
  tensor.matrix<tsl::qint8>()(1, 0) = tsl::qint8(3);
  tensor.matrix<tsl::qint8>()(1, 1) = tsl::qint8(4);

  ::tensorflow::TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  return ::tensorflow::mangling_util::MangleTensor(tensor_proto);
}

std::string GetQint16Tensor() {
  ::tensorflow::Tensor tensor(::tensorflow::DT_QINT16, {2, 2});
  tensor.matrix<tsl::qint16>()(0, 0) = tsl::qint16(1);
  tensor.matrix<tsl::qint16>()(0, 1) = tsl::qint16(2);
  tensor.matrix<tsl::qint16>()(1, 0) = tsl::qint16(3);
  tensor.matrix<tsl::qint16>()(1, 1) = tsl::qint16(4);

  ::tensorflow::TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  return ::tensorflow::mangling_util::MangleTensor(tensor_proto);
}

std::string GetQint32Tensor() {
  ::tensorflow::Tensor tensor(::tensorflow::DT_QINT32, {2, 2});
  tensor.matrix<tsl::qint32>()(0, 0) = tsl::qint32(1);
  tensor.matrix<tsl::qint32>()(0, 1) = tsl::qint32(2);
  tensor.matrix<tsl::qint32>()(1, 0) = tsl::qint32(3);
  tensor.matrix<tsl::qint32>()(1, 1) = tsl::qint32(4);

  ::tensorflow::TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  return ::tensorflow::mangling_util::MangleTensor(tensor_proto);
}

std::unique_ptr<MLIRContext> CreateContext() {
  auto context = std::make_unique<MLIRContext>();
  DialectRegistry mlir_registry;
  RegisterCommonToolingDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);
  context->getOrLoadDialect<tf_type::TFTypeDialect>();
  context->getOrLoadDialect<quant::QuantizationDialect>();
  context->getOrLoadDialect<mlir::mhlo::MhloDialect>();
  context->getOrLoadDialect<sparse_tensor::SparseTensorDialect>();
  return context;
}

TEST(GetDenseAttrFromTensorProtoAttrTest, Qint8ToUQ8Succeeds) {
  auto context = CreateContext();
  TensorType result_tensor_type = RankedTensorType::get(
      {2, 2}, quant::UniformQuantizedType::get(
                  quant::QuantizationFlags::FlagValue::Signed,
                  IntegerType::get(context.get(), 8),
                  FloatType::getF32(context.get()), 3.0, 2, -128, 127));

  auto dense_attr =
      GetDenseAttrFromTensorProtoAttr(GetQint8Tensor(), result_tensor_type);

  ASSERT_TRUE(succeeded(dense_attr));
  EXPECT_THAT(dense_attr->getValues<int8_t>(), testing::SizeIs(4));
  EXPECT_EQ(dense_attr->getValues<int8_t>()[0], 1);
  EXPECT_EQ(dense_attr->getValues<int8_t>()[1], 2);
  EXPECT_EQ(dense_attr->getValues<int8_t>()[2], 3);
  EXPECT_EQ(dense_attr->getValues<int8_t>()[3], 4);
}

TEST(GetDenseAttrFromTensorProtoAttrTest, Qint8ToInt8Succeeds) {
  auto context = CreateContext();
  TensorType result_tensor_type =
      RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 8));

  auto dense_attr =
      GetDenseAttrFromTensorProtoAttr(GetQint8Tensor(), result_tensor_type);

  ASSERT_TRUE(succeeded(dense_attr));
  EXPECT_THAT(dense_attr->getValues<int8_t>(), testing::SizeIs(4));
  EXPECT_EQ(dense_attr->getValues<int8_t>()[0], 1);
  EXPECT_EQ(dense_attr->getValues<int8_t>()[1], 2);
  EXPECT_EQ(dense_attr->getValues<int8_t>()[2], 3);
  EXPECT_EQ(dense_attr->getValues<int8_t>()[3], 4);
}

TEST(GetDenseAttrFromTensorProtoAttrTest, Qint32ToUQ32Succeeds) {
  auto context = CreateContext();
  TensorType result_tensor_type = RankedTensorType::get(
      {2, 2},
      quant::UniformQuantizedType::get(
          quant::QuantizationFlags::FlagValue::Signed,
          IntegerType::get(context.get(), 32), FloatType::getF32(context.get()),
          3.0, 2, -2147483648, 2147483647));

  auto dense_attr =
      GetDenseAttrFromTensorProtoAttr(GetQint32Tensor(), result_tensor_type);

  ASSERT_TRUE(succeeded(dense_attr));
  EXPECT_THAT(dense_attr->getValues<int32_t>(), testing::SizeIs(4));
  EXPECT_EQ(dense_attr->getValues<int32_t>()[0], 1);
  EXPECT_EQ(dense_attr->getValues<int32_t>()[1], 2);
  EXPECT_EQ(dense_attr->getValues<int32_t>()[2], 3);
  EXPECT_EQ(dense_attr->getValues<int32_t>()[3], 4);
}

TEST(GetDenseAttrFromTensorProtoAttrTest, Qint32ToInt32Succeeds) {
  auto context = CreateContext();
  TensorType result_tensor_type =
      RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 32));

  auto dense_attr =
      GetDenseAttrFromTensorProtoAttr(GetQint32Tensor(), result_tensor_type);

  ASSERT_TRUE(succeeded(dense_attr));
  EXPECT_THAT(dense_attr->getValues<int32_t>(), testing::SizeIs(4));
  EXPECT_EQ(dense_attr->getValues<int32_t>()[0], 1);
  EXPECT_EQ(dense_attr->getValues<int32_t>()[1], 2);
  EXPECT_EQ(dense_attr->getValues<int32_t>()[2], 3);
  EXPECT_EQ(dense_attr->getValues<int32_t>()[3], 4);
}

TEST(GetDenseAttrFromTensorProtoAttrTest, UnsupportedQint16Fails) {
  auto context = CreateContext();
  TensorType result_tensor_type =
      RankedTensorType::get({2, 2}, IntegerType::get(context.get(), 16));

  EXPECT_TRUE(failed(
      GetDenseAttrFromTensorProtoAttr(GetQint16Tensor(), result_tensor_type)));
}

TEST(IsTFQintTypeTest, ValidTFQintTypeSucceeds) {
  auto context = CreateContext();

  EXPECT_TRUE(IsTFQintType(TF::Qint8Type::get(context.get())));
  EXPECT_TRUE(IsTFQintType(TF::Qint16Type::get(context.get())));
  EXPECT_TRUE(IsTFQintType(TF::Qint32Type::get(context.get())));
  EXPECT_TRUE(IsTFQintType(TF::Quint8Type::get(context.get())));
  EXPECT_TRUE(IsTFQintType(TF::Quint16Type::get(context.get())));

  EXPECT_FALSE(IsTFQintType(TF::Int8RefType::get(context.get())));
  EXPECT_FALSE(IsTFQintType(TF::Float8E5M2RefType::get(context.get())));
}

TEST(GetIntTypeFromTFQintTest, ChecksIntTypesFromTFQint) {
  auto context = CreateContext();

  auto type = GetIntTypeFromTFQint(TF::Qint8Type::get(context.get()));
  EXPECT_TRUE(llvm::isa<IntegerType>(type));
  EXPECT_EQ(mlir::dyn_cast<IntegerType>(type).getWidth(), 8);
  EXPECT_FALSE(mlir::dyn_cast<IntegerType>(type).isSigned());
  EXPECT_FALSE(mlir::dyn_cast<IntegerType>(type).isUnsigned());

  type = GetIntTypeFromTFQint(TF::Qint16Type::get(context.get()));
  EXPECT_TRUE(llvm::isa<IntegerType>(type));
  EXPECT_EQ(mlir::dyn_cast<IntegerType>(type).getWidth(), 16);
  EXPECT_FALSE(mlir::dyn_cast<IntegerType>(type).isSigned());
  EXPECT_FALSE(mlir::dyn_cast<IntegerType>(type).isUnsigned());

  type = GetIntTypeFromTFQint(TF::Qint32Type::get(context.get()));
  EXPECT_TRUE(llvm::isa<IntegerType>(type));
  EXPECT_EQ(mlir::dyn_cast<IntegerType>(type).getWidth(), 32);
  EXPECT_FALSE(mlir::dyn_cast<IntegerType>(type).isSigned());
  EXPECT_FALSE(mlir::dyn_cast<IntegerType>(type).isUnsigned());

  type = GetIntTypeFromTFQint(TF::Quint8Type::get(context.get()));
  EXPECT_TRUE(llvm::isa<IntegerType>(type));
  EXPECT_EQ(mlir::dyn_cast<IntegerType>(type).getWidth(), 8);
  EXPECT_TRUE(mlir::dyn_cast<IntegerType>(type).isUnsigned());

  type = GetIntTypeFromTFQint(TF::Quint16Type::get(context.get()));
  EXPECT_TRUE(llvm::isa<IntegerType>(type));
  EXPECT_EQ(mlir::dyn_cast<IntegerType>(type).getWidth(), 16);
  EXPECT_TRUE(mlir::dyn_cast<IntegerType>(type).isUnsigned());

  // Non qint types are returned as is.
  EXPECT_EQ(GetIntTypeFromTFQint(IntegerType::get(type.getContext(), 32)),
            IntegerType::get(type.getContext(), 32));
}

}  // namespace
}  // namespace mlir::quant::tensorflow
