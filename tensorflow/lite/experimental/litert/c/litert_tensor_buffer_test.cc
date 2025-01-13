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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/dmabuf_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/event.h"
#include "tensorflow/lite/experimental/litert/runtime/fastrpc_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/runtime/ion_buffer.h"  // IWYU pragma: keep

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

}  // namespace

TEST(TensorBuffer, HostMemory) {
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kTensorBufferType);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_EQ(tensor_type.layout.strides, nullptr);

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferSize(tensor_buffer, &size), kLiteRtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LiteRtGetTensorBufferOffset(tensor_buffer, &offset),
            kLiteRtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
}

TEST(TensorBuffer, Ahwb) {
  if (!litert::internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeAhwb;

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kTensorBufferType);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_EQ(tensor_type.layout.strides, nullptr);

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferSize(tensor_buffer, &size), kLiteRtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LiteRtGetTensorBufferOffset(tensor_buffer, &offset),
            kLiteRtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
}

TEST(TensorBuffer, Ion) {
  if (!litert::internal::IonBuffer::IsSupported()) {
    GTEST_SKIP()
        << "ION buffers are not supported on this platform; skipping the test";
  }

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeIon;

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kTensorBufferType);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_EQ(tensor_type.layout.strides, nullptr);

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferSize(tensor_buffer, &size), kLiteRtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LiteRtGetTensorBufferOffset(tensor_buffer, &offset),
            kLiteRtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
}

TEST(TensorBuffer, DmaBuf) {
  if (!litert::internal::DmaBufBuffer::IsSupported()) {
    GTEST_SKIP()
        << "DMA-BUF buffers are not supported on this platform; skipping "
           "the test";
  }

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeDmaBuf;

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kTensorBufferType);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_EQ(tensor_type.layout.strides, nullptr);

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferSize(tensor_buffer, &size), kLiteRtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LiteRtGetTensorBufferOffset(tensor_buffer, &offset),
            kLiteRtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
}

TEST(TensorBuffer, FastRpc) {
  if (!litert::internal::FastRpcBuffer::IsSupported()) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeFastRpc;

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kTensorBufferType);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_EQ(tensor_type.layout.strides, nullptr);

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferSize(tensor_buffer, &size), kLiteRtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LiteRtGetTensorBufferOffset(tensor_buffer, &offset),
            kLiteRtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(
      LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, /*event=*/nullptr),
      kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
}

TEST(TensorBuffer, Event) {
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;
  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  bool has_event = true;
  ASSERT_EQ(LiteRtHasTensorBufferEvent(tensor_buffer, &has_event),
            kLiteRtStatusOk);
  EXPECT_FALSE(has_event);

  LiteRtEventT event;
  ASSERT_EQ(LiteRtSetTensorBufferEvent(tensor_buffer, &event), kLiteRtStatusOk);

  has_event = false;
  ASSERT_EQ(LiteRtHasTensorBufferEvent(tensor_buffer, &has_event),
            kLiteRtStatusOk);
  EXPECT_TRUE(has_event);

  LiteRtEvent actual_event;
  ASSERT_EQ(LiteRtGetTensorBufferEvent(tensor_buffer, &actual_event),
            kLiteRtStatusOk);
  ASSERT_EQ(actual_event, &event);

  ASSERT_EQ(LiteRtClearTensorBufferEvent(tensor_buffer), kLiteRtStatusOk);
  ASSERT_EQ(actual_event, &event);

  has_event = true;
  ASSERT_EQ(LiteRtHasTensorBufferEvent(tensor_buffer, &has_event),
            kLiteRtStatusOk);
  EXPECT_FALSE(has_event);

  LiteRtDestroyTensorBuffer(tensor_buffer);
}
