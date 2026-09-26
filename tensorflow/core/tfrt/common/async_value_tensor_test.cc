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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

class FakeOpaqueAllocator : public Allocator {
 public:
  explicit FakeOpaqueAllocator(void* ptr) : ptr_(ptr) {}
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return ptr_;
  }
  void DeallocateRaw(void* ptr) override {}
  bool AllocatesOpaqueHandle() const override { return true; }
  std::string Name() override { return "fake-opaque"; }

 private:
  void* ptr_;
};

TEST(AsyncValueTensorTest, InvalidTensor) {
  tensorflow::Tensor tensor(tensorflow::DT_INT64, tensorflow::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_EQ(avt, nullptr);
}

TEST(AsyncValueTensorTest, SlicedOpaqueTensorReturnsNull) {
  alignas(AsyncValueTensor) char buffer[64];
  FakeOpaqueAllocator allocator(buffer);
  tensorflow::Tensor tensor(&allocator, tensorflow::DT_UINT8,
                            tensorflow::TensorShape({16}));
  tensorflow::Tensor sliced_tensor = tensor.Slice(1, 16);

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&sliced_tensor);

  EXPECT_EQ(avt, nullptr);
}

TEST(AsyncValueTensorTest, UnalignedOpaquePointerReturnsNull) {
  alignas(AsyncValueTensor) char buffer[64];
  FakeOpaqueAllocator allocator(buffer + 3);
  tensorflow::Tensor tensor(&allocator, tensorflow::DT_UINT8,
                            tensorflow::TensorShape({16}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  EXPECT_EQ(avt, nullptr);
}

TEST(AsyncValueTensorTest, FromOpaquePointer) {
  EXPECT_EQ(AsyncValueTensor::FromOpaquePointer(nullptr), nullptr);

  // Tagged pointer with zero raw address -> returns nullptr
  EXPECT_EQ(AsyncValueTensor::FromOpaquePointer(reinterpret_cast<void*>(1)),
            nullptr);

  alignas(AsyncValueTensor) char buffer[sizeof(AsyncValueTensor) + 16];
  void* aligned_ptr = buffer;

  // Untagged aligned pointer -> returns nullptr (tag bit 0 is not set)
  EXPECT_EQ(AsyncValueTensor::FromOpaquePointer(aligned_ptr), nullptr);

  // Tagged unaligned pointer -> returns nullptr
  void* unaligned_tagged_ptr =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(aligned_ptr) + 3);
  EXPECT_EQ(AsyncValueTensor::FromOpaquePointer(unaligned_tagged_ptr), nullptr);

  // Tagged aligned pointer -> returns untagged AsyncValueTensor*
  void* aligned_tagged_ptr =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(aligned_ptr) | 1ULL);
  AsyncValueTensor* expected = reinterpret_cast<AsyncValueTensor*>(aligned_ptr);
  EXPECT_EQ(AsyncValueTensor::FromOpaquePointer(aligned_tagged_ptr), expected);
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

}  // namespace
}  // namespace tensorflow
