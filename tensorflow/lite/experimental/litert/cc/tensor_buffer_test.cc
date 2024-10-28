// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/core/ahwb_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/core/dmabuf_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/core/fastrpc_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/core/ion_buffer.h"  // IWYU pragma: keep

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    /*.layout=*/{
        /*.rank=*/1,
        /*.dimensions=*/kTensorDimensions,
        /*.strides=*/nullptr,
    }};
}  // namespace

TEST(TensorBuffer, HostMemory) {
  const litert::RankedTensorType kTensorType(::kTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer.ok());

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type.ok());
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type.ok());

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size.ok());
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset.ok());
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, Ahwb) {
  if (!litert::internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }

  const litert::RankedTensorType kTensorType(::kTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeAhwb;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer.ok());

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type.ok());
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type.ok());

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size.ok());
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset.ok());
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, Ion) {
  if (!litert::internal::IonBuffer::IsSupported()) {
    GTEST_SKIP()
        << "ION buffers are not supported on this platform; skipping the test";
  }

  const litert::RankedTensorType kTensorType(::kTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeIon;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer.ok());

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type.ok());
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type.ok());

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size.ok());
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset.ok());
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, DmaBuf) {
  if (!litert::internal::DmaBufBuffer::IsSupported()) {
    GTEST_SKIP()
        << "DMA-BUF buffers are not supported on this platform; skipping "
           "the test";
  }

  const litert::RankedTensorType kTensorType(::kTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeDmaBuf;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer.ok());

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type.ok());
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type.ok());

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size.ok());
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset.ok());
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, FastRpc) {
  if (!litert::internal::FastRpcBuffer::IsSupported()) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }

  const litert::RankedTensorType kTensorType(::kTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeFastRpc;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer.ok());

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type.ok());
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type.ok());

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size.ok());
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset.ok());
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr.ok());
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, NotOwned) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(kLiteRtTensorBufferTypeHostMemory,
                                            &kTensorType, sizeof(kTensorData),
                                            &litert_tensor_buffer),
            kLiteRtStatusOk);

  litert::TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/false);
  ASSERT_EQ(tensor_buffer.Get(), litert_tensor_buffer);

  LiteRtDestroyTensorBuffer(litert_tensor_buffer);
}
