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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/dmabuf_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/fastrpc_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/ion_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace {

using ::testing::Eq;
using ::testing::Ne;

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr int kFakeSyncFenceFd = 1;

constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    BuildLayout(kTensorDimensions)};

int GetReferenceCount(const TensorBuffer& tensor_buffer) {
  LiteRtTensorBufferT* internal_tensor_buffer =
      static_cast<LiteRtTensorBufferT*>(tensor_buffer.Get());
  return internal_tensor_buffer->RefCount();
}

TEST(TensorBuffer, HostMemory) {
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
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
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, Ahwb) {
  if (!internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeAhwb;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
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
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, Ion) {
  if (!internal::IonBuffer::IsSupported()) {
    GTEST_SKIP()
        << "ION buffers are not supported on this platform; skipping the test";
  }

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeIon;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
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
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, DmaBuf) {
  if (!internal::DmaBufBuffer::IsSupported()) {
    GTEST_SKIP()
        << "DMA-BUF buffers are not supported on this platform; skipping "
           "the test";
  }

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeDmaBuf;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
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
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, FastRpc) {
  if (!internal::FastRpcBuffer::IsSupported()) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeFastRpc;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
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
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(*tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, NotOwned) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                kLiteRtTensorBufferTypeHostMemory, &kTestTensorType,
                sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/false);
  ASSERT_EQ(tensor_buffer.Get(), litert_tensor_buffer);

  LiteRtDestroyTensorBuffer(litert_tensor_buffer);
}

TEST(TensorBuffer, CreateFromExternalHostMemory) {
  // Allocate a tensor buffer with host memory.
  const int kTensorBufferSize =
      std::max<int>(sizeof(kTensorData), LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  const RankedTensorType kTensorType(kTestTensorType);
  void* host_memory_ptr;
  ASSERT_EQ(
      ::posix_memalign(&host_memory_ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                       kTensorBufferSize),
      0);

  std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));

  // Create a tensor buffer that wraps the host memory.
  auto tensor_buffer_from_external_memory = TensorBuffer::CreateFromHostMemory(
      kTensorType, host_memory_ptr, kTensorBufferSize);

  auto lock_and_addr_external_memory =
      TensorBufferScopedLock::Create(*tensor_buffer_from_external_memory);
  ASSERT_TRUE(lock_and_addr_external_memory);
  ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                        sizeof(kTensorData)),
            0);

  free(host_memory_ptr);
}

#if LITERT_HAS_AHWB_SUPPORT
TEST(TensorBuffer, CreateFromAhwb) {
  AHardwareBuffer* ahw_buffer = nullptr;
  if (__builtin_available(android 26, *)) {
    int error = 0;
    AHardwareBuffer_Desc desc = {
        .width = LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY |
                 AHARDWAREBUFFER_USAGE_CPU_READ_RARELY};
    error = AHardwareBuffer_allocate(&desc, &ahw_buffer);
    ASSERT_EQ(error, 0);

    void* host_memory_ptr = nullptr;
    error =
        AHardwareBuffer_lock(ahw_buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                             -1, nullptr, &host_memory_ptr);
    ASSERT_EQ(error, 0);

    std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));

    int fence_file_descriptor = -1;
    error = AHardwareBuffer_unlock(ahw_buffer, &fence_file_descriptor);
    ASSERT_EQ(error, 0);
  } else {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }

  {
    // Create a tensor buffer that wraps the AHardwareBuffer.
    const RankedTensorType kTensorType(kTestTensorType);
    auto tensor_buffer_from_ahwb =
        TensorBuffer::CreateFromAhwb(kTensorType, ahw_buffer,
                                     /*ahwb_offset=*/0);

    auto lock_and_addr_external_memory =
        TensorBufferScopedLock::Create(*tensor_buffer_from_ahwb);
    ASSERT_TRUE(lock_and_addr_external_memory);
    ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                          sizeof(kTensorData)),
              0);
  }

  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_release(ahw_buffer);
  }
}
#endif  // LITERT_HAS_AHWB_SUPPORT

TEST(TensorBuffer, Duplicate) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                kLiteRtTensorBufferTypeHostMemory, &kTestTensorType,
                sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/true);
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
        TensorBufferScopedLock::Create(*duplicated_tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));

    // When the scope ends, the duplicated tensor buffer should be destroyed.
    // This should not affect the original tensor buffer.
  }

  ASSERT_EQ(GetReferenceCount(tensor_buffer), 1);
  // Check that the original tensor buffer is not affected.
  {
    auto lock_and_addr = TensorBufferScopedLock::Create(tensor_buffer);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, ReadWriteBasic) {
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                kLiteRtTensorBufferTypeHostMemory, &kTestTensorType,
                sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer(litert_tensor_buffer, /*owned=*/true);
  auto write_success = tensor_buffer.Write<float>(absl::MakeSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0])));
  ASSERT_TRUE(write_success);
  float read_data[sizeof(kTensorData) / sizeof(kTensorData[0])];
  auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  ASSERT_EQ(std::memcmp(read_data, kTensorData, sizeof(kTensorData)), 0);
}

TEST(TensorBuffer, ReadWriteBufferSizeMismatch) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeHostMemory,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
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

#if LITERT_HAS_OPENGL_SUPPORT
TEST(TensorBuffer, CreateFromGlTexture) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  // Create GL texture.
  tflite::gpu::gl::GlTexture gl_texture(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1,
                                        /*has_ownership=*/true);
  ASSERT_TRUE(gl_texture.is_valid());

  // Create tensor buffer from existing GL texture (e.g. this could be from
  // Android Camera API).
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateFromGlTexture(
          RankedTensorType(kTensorType), gl_texture.target(), gl_texture.id(),
          gl_texture.format(), gl_texture.bytes_size(), gl_texture.layer()));
}

tflite::gpu::gl::GlBuffer CreateTestGlBuffer(size_t size_bytes) {
  tflite::gpu::gl::GlBuffer gl_buffer;
  CHECK_OK(tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<std::byte>(
      size_bytes, &gl_buffer));
  return gl_buffer;
}

TEST(TensorBuffer, CreateFromGlBuffer) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  // Create GL buffer.
  tflite::gpu::gl::GlBuffer gl_buffer = CreateTestGlBuffer(sizeof(kTensorData));

  // Create tensor buffer from existing GL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateFromGlBuffer(
          RankedTensorType(kTensorType), gl_buffer.target(), gl_buffer.id(),
          gl_buffer.bytes_size(), gl_buffer.offset()));
}

#if LITERT_HAS_AHWB_SUPPORT
TEST(TensorBuffer, GetGlBufferFromAhwb) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  // Create AHWB Tensor buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ahwb_tensor_buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeAhwb,
                                  RankedTensorType(kTensorType),
                                  sizeof(kTensorData)));

  // Write to AHWB Tensor buffer.
  LITERT_ASSERT_OK(ahwb_tensor_buffer.Write<float>(absl::MakeConstSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0]))));

  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer::GlBuffer gl_buffer,
                              ahwb_tensor_buffer.GetGlBuffer());
  EXPECT_THAT(gl_buffer.target, Eq(GL_SHADER_STORAGE_BUFFER));
  EXPECT_THAT(gl_buffer.id, Ne(0));
  EXPECT_THAT(gl_buffer.size_bytes, Eq(sizeof(kTensorData)));
  EXPECT_THAT(gl_buffer.offset, Eq(0));

  // Read from GL buffer.
  // TODO(gcarranza): Add GlBuffer ReadLock functionality to LiteRT
  // TensorBuffer. GlBuffer::Unlock currently writes to GL buffer.
  tflite::gpu::gl::GlBuffer gl_buffer_from_ahwb(
      gl_buffer.target, gl_buffer.id, gl_buffer.size_bytes, gl_buffer.offset,
      /*has_ownership=*/false);
  float read_data[sizeof(kTensorData) / sizeof(kTensorData[0])];
  ASSERT_OK(gl_buffer_from_ahwb.Read<float>(absl::MakeSpan(read_data)));
  ASSERT_EQ(std::memcmp(read_data, kTensorData, sizeof(kTensorData)), 0);
}
#endif  // LITERT_HAS_AHWB_SUPPORT

#endif  // LITERT_HAS_OPENGL_SUPPORT

TEST(TensorBuffer, GetAhwb) {
  if (!internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeAhwb,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(AHardwareBuffer * ahwb, tensor_buffer.GetAhwb());
  EXPECT_THAT(ahwb, Ne(nullptr));
}

TEST(TensorBuffer, Event) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeHostMemory,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  // Create event.
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateFromSyncFenceFd(kFakeSyncFenceFd, true));
  // Move event into tensor buffer.
  LITERT_EXPECT_OK(tensor_buffer.SetEvent(std::move(event)));
  EXPECT_TRUE(tensor_buffer.HasEvent());
  LITERT_ASSERT_OK_AND_ASSIGN(Event tensor_buffer_event,
                              tensor_buffer.GetEvent());
  LITERT_ASSERT_OK_AND_ASSIGN(int fence_fd,
                              tensor_buffer_event.GetSyncFenceFd());
  EXPECT_THAT(fence_fd, Eq(kFakeSyncFenceFd));
  // Clear event.
  LITERT_ASSERT_OK(tensor_buffer.ClearEvent());
  EXPECT_FALSE(tensor_buffer.HasEvent());
}

}  // namespace
}  // namespace litert
