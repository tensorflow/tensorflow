/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/mock-log.h"
#include "absl/base/log_severity.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::HasSubstr;
using ::testing::kDoNotCaptureLogsYet;
using ::testing::ScopedMockLog;

TEST(IfrtRestoreTensorRegistryTest, RetrieveNonRegisteredTensorFails) {
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.GetRestoredTensor("input_tensor_1").Await(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(IfrtRestoreTensorRegistryTest,
     RetrieveNonRegisteredTensorDTypeAndShapeFails) {
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.GetDtypeAndShape("input_tensor_1"),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(IfrtRestoreTensorRegistryTest, SetNonExistedTensorAsUsedByHostFails) {
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.SetUsedByHost("input_tensor_1"),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(IfrtRestoreTensorRegistryTest,
     RegisteredExistedTensorSucceedsAndWarningIsLogged) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto [promise, future] = tsl::MakePromise<tensorflow::Tensor>();

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape = tsl::Future<DtypeAndShape>(DtypeAndShape{
          .dtype = DT_INT32,
          .shape = tensorflow::TensorShape({2, 2}),
      }),
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_2", restored_tensor_info),
              absl_testing::IsOk());
  promise.Set(input_tensor);

  ScopedMockLog mock_log(kDoNotCaptureLogsYet);
  EXPECT_CALL(mock_log, Log).Times(AnyNumber());
  EXPECT_CALL(mock_log,
              Log(base_logging::WARNING, _,
                  HasSubstr("Variable named 'input_tensor_2' has been already "
                            "registered. Ignore request of a new tensor")))
      .Times(1);
  mock_log.StartCapturingLogs();

  EXPECT_THAT(registry.TryRegister("input_tensor_2", restored_tensor_info),
              absl_testing::IsOk());
}

TEST(IfrtRestoreTensorRegistryTest, SetTensorAsUsedByHost) {
  auto [promise, future] = tsl::MakePromise<tensorflow::Tensor>();
  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape = tsl::Future<DtypeAndShape>(DtypeAndShape{
          .dtype = DT_INT32,
          .shape = tensorflow::TensorShape({2, 2}),
      }),
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_1", restored_tensor_info),
              absl_testing::IsOk());
  EXPECT_THAT(registry.SetUsedByHost("input_tensor_1"), absl_testing::IsOk());
}

TEST(IfrtRestoreTensorRegistryTest, RegisteredTensorCanBeRetrieved) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto [promise, future] = tsl::MakePromise<tensorflow::Tensor>();

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape = tsl::Future<DtypeAndShape>(DtypeAndShape{
          .dtype = DT_INT32,
          .shape = tensorflow::TensorShape({2, 2}),
      }),
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_1", restored_tensor_info),
              absl_testing::IsOk());
  promise.Set(input_tensor);
  TF_ASSERT_OK_AND_ASSIGN(tensorflow::Tensor retrieved,
                          registry.GetRestoredTensor("input_tensor_1").Await());
  test::ExpectEqual(retrieved, input_tensor);
  TF_ASSERT_OK_AND_ASSIGN(DtypeAndShape dtype_and_shape,
                          registry.GetDtypeAndShape("input_tensor_1"));
  EXPECT_TRUE(
      dtype_and_shape.shape.IsSameSize(tensorflow::TensorShape({2, 2})));
  EXPECT_EQ(dtype_and_shape.dtype, DT_INT32);
}

TEST(IfrtRestoreTensorRegistryTest,
     RegisteredTensorDTypeAndShapeCanBeRetrieved) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto [promise, future] = tsl::MakePromise<tensorflow::Tensor>();

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape = tsl::Future<DtypeAndShape>(DtypeAndShape{
          .dtype = DT_INT32,
          .shape = tensorflow::TensorShape({2, 2}),
      }),
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_1", restored_tensor_info),
              absl_testing::IsOk());
  TF_ASSERT_OK_AND_ASSIGN(DtypeAndShape dtype_and_shape,
                          registry.GetDtypeAndShape("input_tensor_1"));
  EXPECT_TRUE(
      dtype_and_shape.shape.IsSameSize(tensorflow::TensorShape({2, 2})));
  EXPECT_EQ(dtype_and_shape.dtype, DT_INT32);
}

TEST(IfrtRestoreTensorRegistryTest, FeezeTensorRegistry) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto [promise1, future1] = tsl::MakePromise<tensorflow::Tensor>();
  auto [promise2, future2] = tsl::MakePromise<tensorflow::Tensor>();

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info1 = {
      .used_by_host = false,
      .dtype_and_shape = tsl::Future<DtypeAndShape>(DtypeAndShape{
          .dtype = DT_INT32,
          .shape = tensorflow::TensorShape({2, 2}),
      }),
      .tensor_future = future1};
  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info2 = {
      .used_by_host = true,
      .dtype_and_shape = tsl::Future<DtypeAndShape>(DtypeAndShape{
          .dtype = DT_INT32,
          .shape = tensorflow::TensorShape({2, 2}),
      }),
      .tensor_future = future2};
  IfrtRestoreTensorRegistry registry;
  TF_ASSERT_OK(registry.TryRegister("input_tensor_1", restored_tensor_info1));
  TF_ASSERT_OK(registry.TryRegister("input_tensor_2", restored_tensor_info2));
  promise1.Set(input_tensor);
  promise2.Set(input_tensor);
  registry.Freeze();
  // Tensor with `used_by_host` set to false will be freed after freeze.
  EXPECT_THAT(registry.GetRestoredTensor("input_tensor_1").Await(),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));
  // Tensor with `used_by_host` set to true will be kept after freeze.
  TF_ASSERT_OK_AND_ASSIGN(tensorflow::Tensor retrieved,
                          registry.GetRestoredTensor("input_tensor_2").Await());
  test::ExpectEqual(retrieved, input_tensor);
}
}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
