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
#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/future.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

using tsl::testing::IsOk;
using tsl::testing::StatusIs;

namespace tensorflow {
namespace ifrt_serving {
namespace {

TEST(IfrtRestoreTensorRegistryTest, RetrieveNonRegisteredTensorFails) {
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.GetRestoredTensor("input_tensor_1").Await(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(IfrtRestoreTensorRegistryTest,
     RetrieveNonRegisteredTensorDTypeAndShapeFails) {
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.GetDtypeAndShape("input_tensor_1"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(IfrtRestoreTensorRegistryTest, SetNonExistedTensorAsUsedByHostFails) {
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.SetUsedByHost("input_tensor_1"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(IfrtRestoreTensorRegistryTest, RegisteredExistedTensorFails) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto promise = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto future = xla::ifrt::Future<tensorflow::Tensor>(promise);

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape =
          {
              .dtype = DT_INT32,
              .shape = tensorflow::TensorShape({2, 2}),
          },
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_2", restored_tensor_info),
              IsOk());
  promise.Set(input_tensor);
  EXPECT_THAT(registry.TryRegister("input_tensor_2", restored_tensor_info),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(IfrtRestoreTensorRegistryTest, SetTensorAsUsedByHost) {
  auto promise = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto future = xla::ifrt::Future<tensorflow::Tensor>(promise);
  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape =
          {
              .dtype = DT_INT32,
              .shape = tensorflow::TensorShape({2, 2}),
          },
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_1", restored_tensor_info),
              IsOk());
  EXPECT_THAT(registry.SetUsedByHost("input_tensor_1"), IsOk());
}

TEST(IfrtRestoreTensorRegistryTest, RegisteredTensorCanBeRetrieved) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto promise = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto future = xla::ifrt::Future<tensorflow::Tensor>(promise);

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape =
          {
              .dtype = DT_INT32,
              .shape = tensorflow::TensorShape({2, 2}),
          },
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_1", restored_tensor_info),
              IsOk());
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
  auto promise = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto future = xla::ifrt::Future<tensorflow::Tensor>(promise);

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      .used_by_host = false,
      .dtype_and_shape =
          {
              .dtype = DT_INT32,
              .shape = tensorflow::TensorShape({2, 2}),
          },
      .tensor_future = future};
  IfrtRestoreTensorRegistry registry;
  EXPECT_THAT(registry.TryRegister("input_tensor_1", restored_tensor_info),
              IsOk());
  TF_ASSERT_OK_AND_ASSIGN(DtypeAndShape dtype_and_shape,
                          registry.GetDtypeAndShape("input_tensor_1"));
  EXPECT_TRUE(
      dtype_and_shape.shape.IsSameSize(tensorflow::TensorShape({2, 2})));
  EXPECT_EQ(dtype_and_shape.dtype, DT_INT32);
}

TEST(IfrtRestoreTensorRegistryTest, FeezeTensorRegistry) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));
  auto promise1 = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto future1 = xla::ifrt::Future<tensorflow::Tensor>(promise1);
  auto promise2 = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
  auto future2 = xla::ifrt::Future<tensorflow::Tensor>(promise2);

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info1 = {
      .used_by_host = false,
      .dtype_and_shape =
          {
              .dtype = DT_INT32,
              .shape = tensorflow::TensorShape({2, 2}),
          },
      .tensor_future = future1};
  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info2 = {
      .used_by_host = true,
      .dtype_and_shape =
          {
              .dtype = DT_INT32,
              .shape = tensorflow::TensorShape({2, 2}),
          },
      .tensor_future = future2};
  IfrtRestoreTensorRegistry registry;
  TF_ASSERT_OK(registry.TryRegister("input_tensor_1", restored_tensor_info1));
  TF_ASSERT_OK(registry.TryRegister("input_tensor_2", restored_tensor_info2));
  promise1.Set(input_tensor);
  promise2.Set(input_tensor);
  registry.Freeze();
  // Tensor with `used_by_host` set to false will be freed after freeze.
  EXPECT_THAT(registry.GetRestoredTensor("input_tensor_1").Await(),
              StatusIs(absl::StatusCode::kUnavailable));
  // Tensor with `used_by_host` set to true will be kept after freeze.
  TF_ASSERT_OK_AND_ASSIGN(tensorflow::Tensor retrieved,
                          registry.GetRestoredTensor("input_tensor_2").Await());
  test::ExpectEqual(retrieved, input_tensor);
}
}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
