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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

// Enable definition of Eigen::ThreadPoolDevice instead of just declaration.
#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
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
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using ::tensorflow::test::TensorEq;
using ::testing::ElementsAre;

Eigen::ThreadPoolDevice GetThreadPoolDevice() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool = []() {
    return new tsl::thread::ThreadPool(tsl::Env::Default(),
                                       tsl::ThreadOptions(), "IfrtSharding",
                                       kMaxParallelism);
  }();
  return Eigen::ThreadPoolDevice(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);
}

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
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client, &thread_pool_device,
                                   tensorflow::IdentityShapeRepresentationFn());

  auto x = tensorflow::test::AsTensor<int32_t>({1, 2, 3},
                                               tensorflow::TensorShape({1, 3}));
  auto y = tensorflow::test::AsTensor<int32_t>({1, 2, 3},
                                               tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs)));

  const auto expected_out = tensorflow::test::AsTensor<int32_t>(
      {14}, tensorflow::TensorShape({1, 1}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
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
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client, &thread_pool_device,
                                   tensorflow::IdentityShapeRepresentationFn());

  auto x1 = tensorflow::test::AsTensor<int32_t>(
      {1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y1 = tensorflow::test::AsTensor<int32_t>(
      {1, 2, 3}, tensorflow::TensorShape({3, 1}));
  const auto expected_out1 = tensorflow::test::AsTensor<int32_t>(
      {14}, tensorflow::TensorShape({1, 1}));
  std::vector<tensorflow::Tensor> inputs1{x1, y1};

  auto x2 = tensorflow::test::AsTensor<int32_t>(
      {1, 2, 3, 4}, tensorflow::TensorShape({1, 4}));
  auto y2 = tensorflow::test::AsTensor<int32_t>(
      {1, 2, 3, 4}, tensorflow::TensorShape({4, 1}));
  const auto expected_out2 = tensorflow::test::AsTensor<int32_t>(
      {30}, tensorflow::TensorShape({1, 1}));

  std::vector<tensorflow::Tensor> inputs2{x2, y2};

  std::vector<tensorflow::Tensor> outputs1, outputs2;
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK_AND_ASSIGN(outputs1,
                            executable.Execute(absl::MakeSpan(inputs1)));
    TF_ASSERT_OK_AND_ASSIGN(outputs2,
                            executable.Execute(absl::MakeSpan(inputs2)));
  }

  ASSERT_EQ(executable.num_executables(), 2);

  EXPECT_THAT(outputs1, ElementsAre(TensorEq(expected_out1)));

  EXPECT_THAT(outputs2, ElementsAre(TensorEq(expected_out2)));
}

TEST(IfrtServingExecutableTest, Spmd) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/spmd_executable.mlir"));

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
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client, &thread_pool_device,
                                   tensorflow::IdentityShapeRepresentationFn());

  auto x = tensorflow::test::AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               tensorflow::TensorShape({4, 2}));
  auto y = tensorflow::test::AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                                               tensorflow::TensorShape({4, 2}));

  auto z = tensorflow::test::AsTensor<int32_t>({21, 22, 23, 24, 25, 26, 27, 28},
                                               tensorflow::TensorShape({4, 2}));

  const auto expected_out = tensorflow::test::AsTensor<int32_t>(
      {33, 36, 39, 42, 45, 48, 51, 54}, tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y, z};
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs)));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST(IfrtServingExecutableTest, SpmdTwoReturns) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/spmd_executable_two_returns.mlir"));

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
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client, &thread_pool_device,
                                   tensorflow::IdentityShapeRepresentationFn());

  auto x = tensorflow::test::AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               tensorflow::TensorShape({4, 2}));
  auto y = tensorflow::test::AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                                               tensorflow::TensorShape({4, 2}));

  auto z = tensorflow::test::AsTensor<int32_t>({21, 22, 23, 24, 25, 26, 27, 28},
                                               tensorflow::TensorShape({4, 2}));

  const auto expected_out0 = tensorflow::test::AsTensor<int32_t>(
      {33, 36, 39, 42, 45, 48, 51, 54}, tensorflow::TensorShape({4, 2}));
  const auto expected_out1 = tensorflow::test::AsTensor<int32_t>(
      {20, 20, 20, 20, 20, 20, 20, 20}, tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y, z};
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs)));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(expected_out0), TensorEq(expected_out1)));
}

TEST(IfrtServingExecutableTest, NoReturn) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/executable_no_return.mlir"));

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
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();

  IfrtServingExecutable executable("test", "main", std::move(mlir_module),
                                   client, &thread_pool_device,
                                   tensorflow::IdentityShapeRepresentationFn());

  auto x = tensorflow::test::AsTensor<int32_t>({1, 2, 3},
                                               tensorflow::TensorShape({1, 3}));
  auto y = tensorflow::test::AsTensor<int32_t>({1, 2, 3},
                                               tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs)));

  ASSERT_EQ(result.size(), 0);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
