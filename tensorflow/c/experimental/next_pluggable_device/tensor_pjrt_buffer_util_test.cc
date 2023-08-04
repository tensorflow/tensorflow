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
#include "tensorflow/c/experimental/next_pluggable_device/tensor_pjrt_buffer_util.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_cpu.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_c_api_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;
using ::testing::NotNull;
using ::tsl::testing::StatusIs;

TEST(TensorPjRtBufferUtilTest, GetPjRtCBufferFromTensorNoBuffer) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  tensorflow::Tensor tensor(allocator.get(), DT_FLOAT, {1});

  EXPECT_THAT(
      GetPjRtCBufferFromTensor(&tensor),
      StatusIs(error::INTERNAL, HasSubstr(absl::StrCat(
                                    "Input tensor does not have PjRtBuffer"))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCBufferFromTensorIncoorectType) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  tensorflow::Tensor tensor(allocator.get(), DT_FLOAT, {1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1));
  std::vector<int32_t> data(1, 0);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      pjrt_client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, pjrt_client->addressable_devices()[0]));
  tensorflow::AsyncValueTensor* av_tensor =
      tensorflow::AsyncValueTensor::FromTensor(&tensor);
  av_tensor->SetBuffer(std::move(buffer));

  EXPECT_THAT(
      GetPjRtCBufferFromTensor(&tensor),
      StatusIs(
          error::INTERNAL,
          HasSubstr(absl::StrCat(
              "The PjRtBuffer in the tensor is not type PjRtCApiBuffer"))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCBufferFromTensorSuccess) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  tensorflow::Tensor tensor(allocator.get(), DT_FLOAT, {1});
  auto status = pjrt::PjrtApi(DEVICE_CPU);
  if (!status.ok()) {
    TF_ASSERT_OK(pjrt::SetPjrtApi(DEVICE_CPU, GetPjrtApi()));
  }
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  std::vector<int32_t> data(1, 0);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      pjrt_client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, pjrt_client->addressable_devices()[0]));
  tensorflow::AsyncValueTensor* av_tensor =
      tensorflow::AsyncValueTensor::FromTensor(&tensor);
  av_tensor->SetBuffer(std::move(buffer));

  TF_ASSERT_OK_AND_ASSIGN(auto c_buffer, GetPjRtCBufferFromTensor(&tensor));

  EXPECT_THAT(c_buffer, NotNull());
}

TEST(TensorPjRtBufferUtilTest, SetPjRtCBufferToTensorNotAsyncValueTensor) {
  tensorflow::Tensor tensor(DT_FLOAT, {1});

  EXPECT_THAT(
      SetPjRtCBufferToTensor(nullptr, nullptr, &tensor),
      StatusIs(
          error::INTERNAL,
          HasSubstr(absl::StrCat(
              "The tensor to set PjRtBuffer is not an AsyncValueTensor"))));
}

TEST(TensorPjRtBufferUtilTest, SetPjRtCBufferToTensorSuccess) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  auto status = pjrt::PjrtApi(DEVICE_CPU);
  if (!status.ok()) {
    TF_ASSERT_OK(pjrt::SetPjrtApi(DEVICE_CPU, GetPjrtApi()));
  }
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  auto c_api_client = down_cast<xla::PjRtCApiClient*>(pjrt_client.get());
  std::vector<int32_t> data(1, 0);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      c_api_client->pjrt_c_client()->client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr,
          c_api_client->pjrt_c_client()->client->addressable_devices()[0]));
  tensorflow::Tensor tensor(allocator.get(), DT_FLOAT, {1});
  auto c_buffer =
      new PJRT_Buffer{std::move(buffer), c_api_client->pjrt_c_client()};

  TF_EXPECT_OK(SetPjRtCBufferToTensor(
      c_buffer, down_cast<xla::PjRtCApiClient*>(pjrt_client.get()), &tensor));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCApiClientNotFound) {
  EXPECT_THAT(
      GetPjRtCApiClient(tensorflow::DeviceType(DEVICE_CPU)),
      StatusIs(error::NOT_FOUND,
               HasSubstr(absl::StrCat("PjRt client not found for device type ",
                                      DEVICE_CPU))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCApiClientIncorrectType) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1));
  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(DEVICE_CPU,
                                                      std::move(pjrt_client)));

  EXPECT_THAT(GetPjRtCApiClient(tensorflow::DeviceType(DEVICE_CPU)),
              StatusIs(error::INTERNAL,
                       HasSubstr(absl::StrCat("PjRtClient for ", DEVICE_CPU,
                                              " is not type PjRtCApiClient"))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCApiClientSuccess) {
  auto status = pjrt::PjrtApi(DEVICE_CPU);
  if (!status.ok()) {
    TF_ASSERT_OK(pjrt::SetPjrtApi(DEVICE_CPU, GetPjrtApi()));
  }
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(DEVICE_CPU,
                                                      std::move(pjrt_client)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client_get,
      GetPjRtCApiClient(tensorflow::DeviceType(DEVICE_CPU)));

  EXPECT_THAT(pjrt_client_get, NotNull());
}

}  // namespace
}  // namespace tensorflow
