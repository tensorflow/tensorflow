/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/attribute/attribute.h"

#include <array>
#include <cstring>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf_mlrt {
namespace {

TEST(AttributeTest, TensorAttr) {
  mlir::MLIRContext mlir_context;
  mlir::Builder builder(&mlir_context);

  std::array<int64_t, 4> data = {0, 1, 2, 3};

  auto dense_i64_attr = builder.getI64VectorAttr(data);

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);
  TF_ASSERT_OK_AND_ASSIGN(
      auto attr_buffer,
      EncodeTensorflowAttribute(emitter_context, dense_i64_attr));

  TensorAttr tensor_attr(attr_buffer.data());

  EXPECT_EQ(tensor_attr.dtype(), tensorflow::DT_INT64);
  EXPECT_THAT(tensor_attr.shape(), ::testing::ElementsAreArray({4}));
  EXPECT_EQ(
      absl::string_view(tensor_attr.data().data(), tensor_attr.data().size()),
      absl::string_view(reinterpret_cast<const char*>(data.data()),
                        data.size() * sizeof(int64_t)));
}

TEST(AttributeTest, BoolTensorAttr) {
  mlir::MLIRContext mlir_context;
  mlir::Builder builder(&mlir_context);

  auto dense_bool_attr = builder.getBoolVectorAttr({true, false, true, false});

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);
  TF_ASSERT_OK_AND_ASSIGN(
      auto attr_buffer,
      EncodeTensorflowAttribute(emitter_context, dense_bool_attr));

  TensorAttr tensor_attr(attr_buffer.data());

  EXPECT_EQ(tensor_attr.dtype(), tensorflow::DT_BOOL);
  EXPECT_THAT(tensor_attr.shape(), ::testing::ElementsAreArray({4}));

  std::array<uint8_t, 4> expected_data = {1, 0, 1, 0};

  EXPECT_EQ(
      absl::string_view(tensor_attr.data().data(), tensor_attr.data().size()),
      absl::string_view(reinterpret_cast<const char*>(expected_data.data()),
                        expected_data.size() * sizeof(uint8_t)));
}

TEST(AttributeTest, SplatTensorAttr) {
  mlir::MLIRContext mlir_context;
  mlir::Builder builder(&mlir_context);

  auto dense_splat_i64_attr = mlir::DenseElementsAttr::get<int64_t>(
      mlir::RankedTensorType::get(4, builder.getI64Type()), 100);

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);
  TF_ASSERT_OK_AND_ASSIGN(
      auto attr_buffer,
      EncodeTensorflowAttribute(emitter_context, dense_splat_i64_attr));

  TensorAttr tensor_attr(attr_buffer.data());

  EXPECT_EQ(tensor_attr.dtype(), tensorflow::DT_INT64);
  EXPECT_THAT(tensor_attr.shape(), ::testing::ElementsAreArray({4}));
  EXPECT_EQ(tensor_attr.data().size(), 4 * sizeof(int64_t));

  const char* p = tensor_attr.data().data();
  for (int i = 0; i < 4; ++i, p += sizeof(int64_t)) {
    int64_t v;
    std::memcpy(&v, p, sizeof(int64_t));
    EXPECT_EQ(v, 100);
  }
}

TEST(AttributeTest, TypedAttr) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::TF::TensorFlowDialect>();
  mlir::Builder builder(&mlir_context);

  auto type_attr = mlir::TypeAttr::get(builder.getType<mlir::IntegerType>(32));

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);
  TF_ASSERT_OK_AND_ASSIGN(
      auto attr_buffer, EncodeTensorflowAttribute(emitter_context, type_attr));
  tensorflow::DataType dtype;
  std::memcpy(&dtype, attr_buffer.data(), sizeof(dtype));

  EXPECT_EQ(dtype, DT_INT32);
}

TEST(AttributeTest, ShapeAttr) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::TF::TensorFlowDialect>();

  std::array<int64_t, 4> data = {1, 2, 3, 4};

  auto shape_attr = mlir::TF::ShapeAttr::get(
      &mlir_context, llvm::ArrayRef<int64_t>(data.begin(), data.end()),
      /*unranked=*/false);

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);
  TF_ASSERT_OK_AND_ASSIGN(
      auto attr_buffer, EncodeTensorflowAttribute(emitter_context, shape_attr));

  ShapeAttr shape_attr_decoded(attr_buffer.data());

  EXPECT_EQ(shape_attr_decoded.unranked(), false);
  EXPECT_THAT(shape_attr_decoded.dims(),
              ::testing::ElementsAreArray({1, 2, 3, 4}));
}

TEST(AttributeTest, DtypeArrayAttr) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::TF::TensorFlowDialect>();
  mlir::Builder builder(&mlir_context);

  std::array<mlir::Attribute, 4> arr = {
      mlir::TypeAttr::get(builder.getType<mlir::IntegerType>(32)),
      mlir::TypeAttr::get(builder.getType<mlir::IntegerType>(64)),
      mlir::TypeAttr::get(builder.getType<mlir::Float32Type>()),
      mlir::TypeAttr::get(builder.getType<mlir::IntegerType>(1))};

  auto arr_attr = mlir::ArrayAttr::get(
      &mlir_context, llvm::ArrayRef<mlir::Attribute>(arr.begin(), arr.end()));

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);
  TF_ASSERT_OK_AND_ASSIGN(auto attr_buffer,
                          EncodeTensorflowAttribute(emitter_context, arr_attr));

  mlrt::bc::Vector<tensorflow::DataType> dtype_arr(attr_buffer.data());
  EXPECT_THAT(dtype_arr, ::testing::ElementsAreArray(
                             {DT_INT32, DT_INT64, DT_FLOAT, DT_BOOL}));
}

TEST(AttributeTest, UnsupportedAttr) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::TF::TensorFlowDialect>();
  mlir::Builder builder(&mlir_context);

  auto dense_string_attr = mlir::DenseStringElementsAttr::get(
      mlir::RankedTensorType::get({2}, builder.getType<mlir::TF::StringType>()),
      {"a", "b"});

  mlrt::AttributeEncoderRegistry attribute_encoder_registry;
  mlrt::ModuleEmitterContext emitter_context(&attribute_encoder_registry);

  EXPECT_THAT(
      EncodeTensorflowAttribute(emitter_context, dense_string_attr),
      ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               "String tensor attribute is not yet supported"));

  EXPECT_THAT(
      EncodeTensorflowAttribute(emitter_context, builder.getUnitAttr()),
      ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               "Try to encode unsupported attribute: unit"));
}

}  // namespace
}  // namespace tf_mlrt
}  // namespace tensorflow
