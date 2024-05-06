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
#include "absl/log/check.h"
#include "absl/status/statusor.h"
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
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/framework/serving_device_selector.h"
#include "tsl/framework/test_util/mock_serving_device_selector.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/platform/tstring.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace {
struct VariableInputTestParam {
  std::vector<tensorflow::Tensor> in_tensors;
  std::vector<bool>
      is_variable;  // if is_variable[i] = true, then in_tensor[i] is a variable
                    // and can be preloaded as an ifrt array.
  std::vector<tensorflow::Tensor> expected_out_tensors;
};
using VariableInputTest = ::testing::TestWithParam<VariableInputTestParam>;

using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;
using ::testing::ElementsAre;
using ::testing::Return;

const tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static auto* const thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

class IfrtServingExecutableTest : public ::testing::Test {
 protected:
  explicit IfrtServingExecutableTest() {
    absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> client =
        xla::ifrt::test_util::GetClient();
    CHECK_OK(client);
    client_ = *std::move(client);
    core_selector_ = std::make_unique<IfrtServingCoreSelector>(&selector_);
  }

  tsl::test_util::MockServingDeviceSelector selector_;
  std::unique_ptr<IfrtServingCoreSelector> core_selector_;
  std::shared_ptr<xla::ifrt::Client> client_;
};

TEST_F(IfrtServingExecutableTest, Basic) {
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

  int64_t program_id = 123456;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillOnce(Return(tsl::DeviceReservation(0, /*selector=*/nullptr)));

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      CreateTfStaticDeviceMgr());

  IfrtServingExecutable executable(
      program_id, "test", "main", std::move(mlir_module), client,
      &GetThreadPool(), &ifrt_loaded_variable_registry,
      &ifrt_restore_tensor_registry, work_queue.get(), device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn(), core_selector_.get());

  auto x = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs), {}));

  const auto expected_out =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST_F(IfrtServingExecutableTest, MultipleShapes) {
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

  int64_t program_id = 123456;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(6)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      CreateTfStaticDeviceMgr());

  IfrtServingExecutable executable(
      program_id, "test", "main", std::move(mlir_module), client,
      &GetThreadPool(), &ifrt_loaded_variable_registry,
      &ifrt_restore_tensor_registry, work_queue.get(), device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn(), core_selector_.get());

  auto x1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  const auto expected_out1 =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));
  std::vector<tensorflow::Tensor> inputs1{x1, y1};

  auto x2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({1, 4}));
  auto y2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({4, 1}));
  const auto expected_out2 =
      AsTensor<int32_t>({30}, tensorflow::TensorShape({1, 1}));

  std::vector<tensorflow::Tensor> inputs2{x2, y2};

  std::vector<tensorflow::Tensor> outputs1, outputs2;
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK_AND_ASSIGN(outputs1,
                            executable.Execute(absl::MakeSpan(inputs1), {}));
    TF_ASSERT_OK_AND_ASSIGN(outputs2,
                            executable.Execute(absl::MakeSpan(inputs2), {}));
  }

  ASSERT_EQ(executable.num_executables(), 2);

  EXPECT_THAT(outputs1, ElementsAre(TensorEq(expected_out1)));

  EXPECT_THAT(outputs2, ElementsAre(TensorEq(expected_out2)));
}

TEST_F(IfrtServingExecutableTest, Spmd) {
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

  int64_t program_id = 111111;

  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id))).Times(0);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      CreateTfStaticDeviceMgr());

  IfrtServingExecutable executable(
      program_id, "test", "main", std::move(mlir_module), client,
      &GetThreadPool(), &ifrt_loaded_variable_registry,
      &ifrt_restore_tensor_registry, work_queue.get(), device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn(), core_selector_.get());

  auto x = AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                             tensorflow::TensorShape({4, 2}));
  auto y = AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                             tensorflow::TensorShape({4, 2}));

  auto z = AsTensor<int32_t>({21, 22, 23, 24, 25, 26, 27, 28},
                             tensorflow::TensorShape({4, 2}));

  const auto expected_out = AsTensor<int32_t>({33, 36, 39, 42, 45, 48, 51, 54},
                                              tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y, z};
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST_F(IfrtServingExecutableTest, SpmdTwoReturns) {
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

  int64_t program_id = 111111;

  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id))).Times(0);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      CreateTfStaticDeviceMgr());

  IfrtServingExecutable executable(
      program_id, "test", "main", std::move(mlir_module), client,
      &GetThreadPool(), &ifrt_loaded_variable_registry,
      &ifrt_restore_tensor_registry, work_queue.get(), device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn(), core_selector_.get());

  auto x = AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                             tensorflow::TensorShape({4, 2}));
  auto y = AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                             tensorflow::TensorShape({4, 2}));

  auto z = AsTensor<int32_t>({21, 22, 23, 24, 25, 26, 27, 28},
                             tensorflow::TensorShape({4, 2}));

  const auto expected_out0 = AsTensor<int32_t>({33, 36, 39, 42, 45, 48, 51, 54},
                                               tensorflow::TensorShape({4, 2}));
  const auto expected_out1 = AsTensor<int32_t>({20, 20, 20, 20, 20, 20, 20, 20},
                                               tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y, z};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(expected_out0), TensorEq(expected_out1)));
}

TEST_F(IfrtServingExecutableTest, NoReturn) {
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

  int64_t program_id = 111111;

  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      CreateTfStaticDeviceMgr());

  IfrtServingExecutable executable(
      program_id, "test", "main", std::move(mlir_module), client,
      &GetThreadPool(), &ifrt_loaded_variable_registry,
      &ifrt_restore_tensor_registry, work_queue.get(), device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn(), core_selector_.get());

  auto x = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute(absl::MakeSpan(inputs), {}));

  ASSERT_EQ(result.size(), 0);
}

TEST_P(VariableInputTest, InterleaveVariable) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/executable_long_inputs.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);

  tsl::test_util::MockServingDeviceSelector device_selector;
  IfrtServingCoreSelector core_selector(&device_selector);
  int64_t program_id = 111111;

  EXPECT_CALL(device_selector, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/4, /*num_blocking_threads=*/4);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      CreateTfStaticDeviceMgr());
  IfrtServingExecutable executable(
      program_id, "test", "main", std::move(mlir_module), client,
      &GetThreadPool(), &ifrt_loaded_variable_registry,
      &ifrt_restore_tensor_registry, work_queue.get(), device_mgr.get(),
      tensorflow::IdentityShapeRepresentationFn(), &core_selector);

  std::vector<tensorflow::Tensor> inputs;
  std::vector<int> loaded_variable_indices;
  for (int i = 0; i < GetParam().in_tensors.size(); i++) {
    if (GetParam().is_variable[i]) {
      auto input_tensor_promise =
          xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
      auto input_tensor_future =
          xla::ifrt::Future<tensorflow::Tensor>(input_tensor_promise);
      IfrtRestoreTensorRegistry::RestoredTensorInfo restore_tensor_info = {
          .dtype_and_shape{.dtype = GetParam().in_tensors[i].dtype(),
                           .shape = GetParam().in_tensors[i].shape()},
          .tensor_future = input_tensor_future};
      std::string variable_name = absl::StrCat("variable_", i);
      ASSERT_OK(ifrt_restore_tensor_registry.TryRegister(variable_name,
                                                         restore_tensor_info));
      loaded_variable_indices.push_back(i);
      input_tensor_promise.Set(GetParam().in_tensors[i]);
      // Use string tensor containing the key (name) in place of variable
      // tensor.
      tensorflow::Tensor key_tensor(tensorflow::DT_STRING, {});
      key_tensor.scalar<tsl::tstring>()() = variable_name;
      inputs.push_back(key_tensor);
    } else {
      inputs.push_back(GetParam().in_tensors[i]);
    }
  }

  ASSERT_EQ(inputs.size(), GetParam().is_variable.size());

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable.Execute(absl::MakeSpan(inputs),
                                      absl::MakeSpan(loaded_variable_indices)));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(GetParam().expected_out_tensors[0]),
                          TensorEq(GetParam().expected_out_tensors[1]),
                          TensorEq(GetParam().expected_out_tensors[2])));
}

INSTANTIATE_TEST_SUITE_P(
    VariableInputTests, VariableInputTest,
    ::testing::ValuesIn<VariableInputTestParam>(
        {
            // Basic case: all variables or all non-variables.
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {true, true, true, true, true},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {false, false, false, false, false},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            // Variable and non-variables are non-interleaved
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {false, false, false, true, true},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {true, true, false, false, false},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            // Variable and non-variables are interleaved
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {true, false, false, true, false},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {false, true, true, false, true},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
        }));

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
