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

#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/dmabuf_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/fastrpc_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/ion_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};
}  // namespace

int GetReferenceCount(const litert::TensorBuffer& tensor_buffer) {
  LiteRtTensorBufferT* internal_tensor_buffer =
      static_cast<LiteRtTensorBufferT*>(tensor_buffer.Get());
  return internal_tensor_buffer->RefCount();
}

TEST(TensorBuffer, HostMemory) {
  const litert::RankedTensorType kTensorType(::kTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
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
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
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
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
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
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
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
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), litert::ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
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

TEST(TensorBuffer, ExternalHostMemory) {
  // Allocate a tensor buffer with host memory.
  const int kTensorBufferSize =
      std::max<int>(sizeof(kTensorData), LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  const litert::RankedTensorType kTensorType(::kTensorType);
  void* host_memory_ptr;
  ASSERT_EQ(
      ::posix_memalign(&host_memory_ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                       kTensorBufferSize),
      0);
  std::unique_ptr<void, void (*)(void*) noexcept> host_memory_ptr_deleter(
      host_memory_ptr, ::free);

  std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));

  // Create a tensor buffer that wraps the host memory.
  auto tensor_buffer_from_external_memory =
      litert::TensorBuffer::CreateFromHostMemory(kTensorType, host_memory_ptr,
                                                 kTensorBufferSize);

  auto lock_and_addr_external_memory = litert::TensorBufferScopedLock::Create(
      *tensor_buffer_from_external_memory);
  ASSERT_TRUE(lock_and_addr_external_memory);
  ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                        sizeof(kTensorData)),
            0);
}

TEST(TensorBuffer, Duplicate) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(kLiteRtTensorBufferTypeHostMemory,
                                            &kTensorType, sizeof(kTensorData),
                                            &litert_tensor_buffer),
            kLiteRtStatusOk);

  litert::TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/true);
  ASSERT_EQ(GetReferenceCount(tensor_buffer), 1);
  {
    auto duplicated_tensor_buffer = tensor_buffer.Duplicate();
    ASSERT_TRUE(duplicated_tensor_buffer);
    ASSERT_EQ(GetReferenceCount(*duplicated_tensor_buffer), 2);
    // The duplicated tensor buffer should point to the same underlying
    // LiteRtTensorBuffer object.
    ASSERT_EQ(duplicated_tensor_buffer->Get(), tensor_buffer.Get());

    // Update tensor buffer using the duplicated tensor buffer.
    auto lock_and_addr =
        litert::TensorBufferScopedLock::Create(*duplicated_tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));

    // When the scope ends, the duplicated tensor buffer should be destroyed.
    // This should not affect the original tensor buffer.
  }

  ASSERT_EQ(GetReferenceCount(tensor_buffer), 1);
  // Check that the original tensor buffer is not affected.
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, ReadWriteBasic) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(kLiteRtTensorBufferTypeHostMemory,
                                            &kTensorType, sizeof(kTensorData),
                                            &litert_tensor_buffer),
            kLiteRtStatusOk);

  litert::TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/true);
  auto write_success = tensor_buffer.Write<float>(absl::MakeSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0])));
  ASSERT_TRUE(write_success);
  float read_data[sizeof(kTensorData) / sizeof(kTensorData[0])];
  auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  ASSERT_EQ(std::memcmp(read_data, kTensorData, sizeof(kTensorData)), 0);
}

TEST(TensorBuffer, ReadWriteBufferSizeMismatch) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(kLiteRtTensorBufferTypeHostMemory,
                                            &kTensorType, sizeof(kTensorData),
                                            &litert_tensor_buffer),
            kLiteRtStatusOk);

  litert::TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/true);
  {
    // Write with smaller size of data.
    auto write_success =
        tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 1));
    ASSERT_TRUE(write_success);
  }
  {
    constexpr const float big_data[] = {10, 20, 30, 40, 50};
    // Write with larger size of data.
    auto write_success =
        tensor_buffer.Write<float>(absl::MakeSpan(big_data, 5));
    ASSERT_FALSE(write_success);
  }
  auto write_success = tensor_buffer.Write<float>(absl::MakeSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0])));
  ASSERT_TRUE(write_success);
  {
    // Read with smaller size of buffer.
    float read_data[1];
    auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data, 1));
    ASSERT_TRUE(read_success);
    ASSERT_EQ(read_data[0], kTensorData[0]);
  }
  {
    // Read with larger size of buffer.
    float read_data[5];
    auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data, 5));
    ASSERT_FALSE(read_success);
  }
}
