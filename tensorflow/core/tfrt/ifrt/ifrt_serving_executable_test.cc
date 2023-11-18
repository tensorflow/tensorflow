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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

TEST(IfrtServingExecutableTest, Basic) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/executable.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client,
                                   tensorflow::IdentityShapeRepresentationFn());

  tensorflow::Tensor x(tensorflow::DT_INT32, tensorflow::TensorShape({1, 3}));
  tensorflow::Tensor y(tensorflow::DT_INT32, tensorflow::TensorShape({3, 1}));
  for (int i = 0; i < 3; ++i) {
    x.flat<int32_t>()(i) = i + 1;
    y.flat<int32_t>()(i) = i + 1;
  }

  std::vector<tensorflow::Tensor> inputs{x, y};
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs)));

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].dtype(), tensorflow::DT_INT32);
  ASSERT_EQ(result[0].shape(), tensorflow::TensorShape({1, 1}));
  ASSERT_EQ(result[0].flat<int32_t>()(0), 14);
}

TEST(IfrtServingExecutableTest, MultipleShapes) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/executable.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client,
                                   tensorflow::IdentityShapeRepresentationFn());

  constexpr int kDim1 = 3;
  tensorflow::Tensor x1(tensorflow::DT_INT32,
                        tensorflow::TensorShape({1, kDim1}));
  tensorflow::Tensor y1(tensorflow::DT_INT32,
                        tensorflow::TensorShape({kDim1, 1}));
  for (int i = 0; i < kDim1; ++i) {
    x1.flat<int32_t>()(i) = i + 1;
    y1.flat<int32_t>()(i) = i + 1;
  }
  std::vector<tensorflow::Tensor> inputs1{x1, y1};

  constexpr int kDim2 = 4;
  tensorflow::Tensor x2(tensorflow::DT_INT32,
                        tensorflow::TensorShape({1, kDim2}));
  tensorflow::Tensor y2(tensorflow::DT_INT32,
                        tensorflow::TensorShape({kDim2, 1}));
  for (int i = 0; i < kDim2; ++i) {
    x2.flat<int32_t>()(i) = i + 1;
    y2.flat<int32_t>()(i) = i + 1;
  }
  std::vector<tensorflow::Tensor> inputs2{x2, y2};

  std::vector<tensorflow::Tensor> outputs1, outputs2;
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK_AND_ASSIGN(outputs1,
                            executable.Execute(absl::MakeSpan(inputs1)));
    TF_ASSERT_OK_AND_ASSIGN(outputs2,
                            executable.Execute(absl::MakeSpan(inputs2)));
  }
  ASSERT_EQ(outputs1.size(), 1);
  ASSERT_EQ(outputs1[0].dtype(), tensorflow::DT_INT32);
  ASSERT_EQ(outputs1[0].shape(), tensorflow::TensorShape({1, 1}));
  ASSERT_EQ(outputs1[0].flat<int32_t>()(0), 14);

  ASSERT_EQ(outputs2.size(), 1);
  ASSERT_EQ(outputs2[0].dtype(), tensorflow::DT_INT32);
  ASSERT_EQ(outputs2[0].shape(), tensorflow::TensorShape({1, 1}));
  ASSERT_EQ(outputs2[0].flat<int32_t>()(0), 30);

  ASSERT_EQ(executable.num_executables(), 2);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
