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

#include "tensorflow/core/tfrt/ifrt/undonatable_buffer_converter.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/undonatable_common_pjrt_buffer.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using ::absl_testing::IsOk;
using tensorflow::test::TensorEq;

absl::StatusOr<xla::ifrt::ArrayRef> MakeTestArray(
    xla::ifrt::Client& client, const tensorflow::Tensor& tensor,
    const tsl::thread::ThreadPool& thread_pool) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                      client.LookupDevice(xla::ifrt::DeviceId(0)));
  TF_ASSIGN_OR_RETURN(auto device_list, client.MakeDeviceList({device}));
  TF_ASSIGN_OR_RETURN(
      auto sharding,
      ToIfrtSharding(client, xla::HloSharding::Replicate(), device_list));
  return MakeArrayFromTensor(client, tensor, device_list, std::move(sharding),
                             thread_pool, /*xla_input_layout=*/nullptr);
}

void VerifyUndonatableAndContentsMatch(xla::ifrt::Array* array,
                                       const tensorflow::Tensor& expected) {
  auto* pjrt_array = xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleArray>(array);
  ASSERT_NE(pjrt_array, nullptr);
  ASSERT_EQ(pjrt_array->pjrt_buffers().size(), 1);
  for (const std::shared_ptr<xla::PjRtBuffer>& buffer :
       pjrt_array->pjrt_buffers()) {
    auto* undonatable_buffer =
        dynamic_cast<xla::UndonatableCommonPjRtBuffer*>(buffer.get());
    ASSERT_NE(undonatable_buffer, nullptr);
    // ToLiteral/CopyToHostBuffer are Unimplemented on the undonatable buffer;
    // verify contents through the raw-buffer path. The raw buffer reference
    // must stay alive until the copy completes.
    ASSERT_THAT(undonatable_buffer->GetReadyFuture().Await(), IsOk());
    tensorflow::Tensor host_tensor(expected.dtype(), expected.shape());
    auto raw_buffer = undonatable_buffer->AcquireRawBufferRef("ConverterTest");
    ASSERT_THAT(
        raw_buffer
            ->CopyRawDeviceToHost(host_tensor.data(), /*offset=*/0,
                                  /*transfer_size=*/host_tensor.TotalBytes())
            .Await(),
        IsOk());
    EXPECT_THAT(host_tensor, TensorEq(expected));
  }
}

TEST(UndonatableBufferConverterTest, ConvertsBuffersAndPreservesContents) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({2, 2}));
  auto client = xla::ifrt::test_util::GetClient();
  ASSERT_THAT(client, IsOk());
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Converter", /*num_threads=*/4);
  auto array = MakeTestArray(**client, input_tensor, thread_pool);
  ASSERT_THAT(array, IsOk());

  ASSERT_THAT(MakeArrayBuffersUndonatable(array->get()), IsOk());
  VerifyUndonatableAndContentsMatch(array->get(), input_tensor);
}

TEST(UndonatableBufferConverterTest, SecondConversionIsANoOp) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({2, 2}));
  auto client = xla::ifrt::test_util::GetClient();
  ASSERT_THAT(client, IsOk());
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Converter", /*num_threads=*/4);
  auto array = MakeTestArray(**client, input_tensor, thread_pool);
  ASSERT_THAT(array, IsOk());

  ASSERT_THAT(MakeArrayBuffersUndonatable(array->get()), IsOk());
  ASSERT_THAT(MakeArrayBuffersUndonatable(array->get()), IsOk());
  VerifyUndonatableAndContentsMatch(array->get(), input_tensor);
}

TEST(UndonatableBufferConverterTest, NullArrayIsAnError) {
  EXPECT_EQ(MakeArrayBuffersUndonatable(nullptr).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
