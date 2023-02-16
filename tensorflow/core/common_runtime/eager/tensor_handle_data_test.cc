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

#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"

#include <utility>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TensorHandleData, TensorAttribute) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  LocalTensorHandleData handle_data(std::move(t));

  const tensorflow::Tensor* ret_tensor;
  TF_EXPECT_OK(handle_data.Tensor(&ret_tensor));

  EXPECT_EQ(ret_tensor->dtype(), DT_UINT16);
  EXPECT_EQ(ret_tensor->dims(), 2);
}

TEST(TensorHandleData, TensorValueAttribute) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  LocalTensorHandleData handle_data(std::move(t));

  tensorflow::TensorValue tensor_value;
  TF_EXPECT_OK(handle_data.TensorValue(&tensor_value));

  EXPECT_EQ(tensor_value.dtype(), DT_UINT16);
}

TEST(TensorHandleData, TensorShapeAttribute) {
  TensorShape shape({2, 2});
  Tensor t(DT_UINT16, shape);
  LocalTensorHandleData handle_data(std::move(t));

  tensorflow::TensorShape tensor_shape;
  TF_EXPECT_OK(handle_data.Shape(&tensor_shape));

  EXPECT_EQ(tensor_shape, shape);
}

TEST(TensorHandleData, NumDimsAttribute) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  LocalTensorHandleData handle_data(std::move(t));

  int num_dims;
  TF_EXPECT_OK(handle_data.NumDims(&num_dims));

  EXPECT_EQ(num_dims, 2);
}

TEST(TensorHandleData, DimAttribute) {
  Tensor t(DT_UINT16, TensorShape({2, 3}));
  LocalTensorHandleData handle_data(std::move(t));

  int64_t dim;
  TF_EXPECT_OK(handle_data.Dim(1, &dim));

  EXPECT_EQ(dim, 3);
}

TEST(TensorHandleData, NumElementsAttribute) {
  Tensor t(DT_UINT16, TensorShape({2, 3}));
  LocalTensorHandleData handle_data(std::move(t));

  int64_t num_elements;
  TF_EXPECT_OK(handle_data.NumElements(&num_elements));

  EXPECT_EQ(num_elements, 6);
}

TEST(TensorHandleData, UnprotectReady) {
  Tensor t(DT_UINT16, TensorShape({2, 3}));
  LocalTensorHandleData handle_data(std::move(t));
  EXPECT_TRUE(handle_data.IsReady());

  TF_EXPECT_OK(handle_data.Unprotect());
}

TEST(TensorHandleData, UnprotectNotReady) {
  LocalTensorHandleData handle_data;
  EXPECT_FALSE(handle_data.IsReady());

  EXPECT_THAT(handle_data.Unprotect(),
              tensorflow::testing::StatusIs(tensorflow::error::INTERNAL));
}

TEST(TensorHandleData, DebugString) {
  Tensor t(DT_UINT16, TensorShape({2, 3}));
  LocalTensorHandleData handle_data(std::move(t));

  EXPECT_THAT(handle_data.DebugString(),
              ::testing::HasSubstr("Tensor<type: uint16 shape: [2,3]>"));
}

TEST(TensorHandleData, NonBlockingControlPoisonHandle) {
  Tensor t(DT_UINT16, TensorShape({2, 3}));
  LocalTensorHandleData handle_data(std::move(t));
  TF_EXPECT_OK(handle_data.IsPoisoned());

  tensorflow::Status fake_failure_status(tensorflow::error::ABORTED,
                                         "Fake failure.");
  handle_data.Poison(fake_failure_status);

  // NonBlockingControl can never poison the tensor.
  TF_EXPECT_OK(handle_data.IsPoisoned());
}

TEST(TensorHandleData, BlockingControlPoisonHandle) {
  LocalTensorHandleData handle_data;
  TF_EXPECT_OK(handle_data.IsPoisoned());

  tensorflow::Status fake_failure_status(tensorflow::error::ABORTED,
                                         "Fake failure.");
  handle_data.Poison(fake_failure_status);

  EXPECT_THAT(
      handle_data.IsPoisoned(),
      tensorflow::testing::StatusIs(fake_failure_status.code(),
                                    fake_failure_status.error_message()));
}

TEST(TensorHandleData, BlockingControlSetTensor) {
  Tensor t(DT_UINT16, TensorShape({2, 3}));
  LocalTensorHandleData handle_data;

  TF_EXPECT_OK(handle_data.SetTensor(std::move(t)));

  // Access the underlying tensor to ensure it is currently set.
  int64_t num_elements;
  TF_EXPECT_OK(handle_data.NumElements(&num_elements));

  EXPECT_EQ(num_elements, 6);
}

TEST(TensorHandleData, BlockingControlNotReadyDebugString) {
  LocalTensorHandleData handle_data;

  EXPECT_THAT(handle_data.DebugString(),
              ::testing::HasSubstr("LocalTensorHandleData"));
}

}  // namespace
}  // namespace tensorflow
