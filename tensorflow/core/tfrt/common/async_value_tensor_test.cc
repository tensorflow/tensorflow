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

#include "tensorflow/core/tfrt/common/async_value_tensor.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

TEST(AsyncValueTensorTest, InvalidTensor) {
  tensorflow::Tensor tensor(tensorflow::DT_INT64, tensorflow::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_EQ(avt, nullptr);
}

TEST(AsyncValueTensorTest, SetAndGetAsyncValue) {
  AsyncValueAllocator allocator;
  tensorflow::Tensor tensor(&allocator, tensorflow::DT_INT64,
                            tensorflow::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_NE(avt, nullptr);

  tsl::AsyncValueRef<int32_t> value =
      tsl::MakeConstructedAsyncValueRef<int32_t>(123);

  avt->SetAsyncRef(value.CopyRCRef());

  auto ret_value = avt->GetAsyncRef();
  ASSERT_EQ(ret_value, value.CopyRCRef());
}

TEST(AsyncValueTensorTest, SetAndGetBuffer) {
  AsyncValueAllocator allocator;
  tensorflow::Tensor tensor(&allocator, tensorflow::DT_INT64,
                            tensorflow::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_NE(avt, nullptr);

  std::shared_ptr<xla::PjRtBuffer> buffer;

  avt->SetBuffer(buffer);

  auto ret_buffer = avt->GetBuffer();

  ASSERT_EQ(ret_buffer, buffer);
}

class FakeOpaqueTensorBuffer : public TensorBuffer {
 public:
  explicit FakeOpaqueTensorBuffer(void* ptr) : TensorBuffer(ptr) {}
  size_t size() const override { return 100; }
  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(AllocationDescription* proto) const override {}
  bool AllocatesOpaqueHandle() const override { return true; }
};

TEST(AsyncValueTensorTest, SlicedTensorReturnsNull) {
  tensorflow::Tensor tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({10}));
  tensorflow::Tensor sliced_tensor = tensor.Slice(1, 5);

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&sliced_tensor);
  EXPECT_EQ(avt, nullptr);
}

TEST(AsyncValueTensorTest, SlicedOpaqueTensorBufferReturnsNull) {
  auto* fake_buf = new FakeOpaqueTensorBuffer(reinterpret_cast<void*>(0x1000));
  tensorflow::Tensor parent_tensor(DT_UINT8, TensorShape({100}), fake_buf);
  fake_buf->Unref();
  tensorflow::Tensor sliced_tensor = parent_tensor.Slice(1, 10);

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&sliced_tensor);
  EXPECT_EQ(avt, nullptr);
}

}  // namespace
}  // namespace tensorflow
