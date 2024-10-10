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
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/core/dmabuf_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/lrt/core/fastrpc_buffer.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/lrt/core/ion_buffer.h"  // IWYU pragma: keep

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LrtRankedTensorType kTensorType = {
    /*.element_type=*/kLrtElementTypeFloat32,
    /*.layout=*/{
        /*.rank=*/1,
        /*.dimensions=*/kTensorDimensions,
        /*.strides=*/nullptr,
    }};

}  // namespace

TEST(TensorBuffer, HostMemory) {
  LrtTensorBuffer buffer;
  ASSERT_EQ(
      LrtCreateManagedTensorBuffer(kLrtTensorBufferTypeHostMemory, kTensorType,
                                   sizeof(kTensorData), &buffer),
      kLrtStatusOk);

  LrtTensorBufferType buffer_type;
  ASSERT_EQ(LrtGetTensorBufferType(buffer, &buffer_type), kLrtStatusOk);
  ASSERT_EQ(buffer_type, kLrtTensorBufferTypeHostMemory);

  LrtRankedTensorType tensor_type;
  ASSERT_EQ(LrtGetTensorBufferTensorType(buffer, &tensor_type), kLrtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLrtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_TRUE(!tensor_type.layout.strides);

  size_t size;
  ASSERT_EQ(LrtGetTensorBufferSize(buffer, &size), kLrtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LrtGetTensorBufferOffset(buffer, &offset), kLrtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  LrtDestroyTensorBuffer(buffer);
}

#if LRT_HAS_AHWB_SUPPORT
TEST(TensorBuffer, Ahwb) {
  LrtTensorBuffer buffer;
  ASSERT_EQ(LrtCreateManagedTensorBuffer(kLrtTensorBufferTypeAhwb, kTensorType,
                                         sizeof(kTensorData), &buffer),
            kLrtStatusOk);

  LrtTensorBufferType buffer_type;
  ASSERT_EQ(LrtGetTensorBufferType(buffer, &buffer_type), kLrtStatusOk);
  ASSERT_EQ(buffer_type, kLrtTensorBufferTypeAhwb);

  LrtRankedTensorType tensor_type;
  ASSERT_EQ(LrtGetTensorBufferTensorType(buffer, &tensor_type), kLrtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLrtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_TRUE(!tensor_type.layout.strides);

  size_t size;
  ASSERT_EQ(LrtGetTensorBufferSize(buffer, &size), kLrtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LrtGetTensorBufferOffset(buffer, &offset), kLrtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  LrtDestroyTensorBuffer(buffer);
}
#endif  // LRT_HAS_AHWB_SUPPORT

#if LRT_HAS_ION_SUPPORT
TEST(TensorBuffer, Ion) {
  if (!lrt::internal::IonBuffer::IsSupported()) {
    GTEST_SKIP()
        << "ION buffers are not supported on this platform; skipping the test";
  }

  LrtTensorBuffer buffer;
  ASSERT_EQ(LrtCreateManagedTensorBuffer(kLrtTensorBufferTypeIon, kTensorType,
                                         sizeof(kTensorData), &buffer),
            kLrtStatusOk);

  LrtTensorBufferType buffer_type;
  ASSERT_EQ(LrtGetTensorBufferType(buffer, &buffer_type), kLrtStatusOk);
  ASSERT_EQ(buffer_type, kLrtTensorBufferTypeIon);

  LrtRankedTensorType tensor_type;
  ASSERT_EQ(LrtGetTensorBufferTensorType(buffer, &tensor_type), kLrtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLrtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_TRUE(!tensor_type.layout.strides);

  size_t size;
  ASSERT_EQ(LrtGetTensorBufferSize(buffer, &size), kLrtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LrtGetTensorBufferOffset(buffer, &offset), kLrtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  LrtDestroyTensorBuffer(buffer);
}
#endif  // LRT_HAS_ION_SUPPORT

#if LRT_HAS_DMABUF_SUPPORT
TEST(TensorBuffer, DmaBuf) {
  if (!lrt::internal::DmaBufBuffer::IsSupported()) {
    GTEST_SKIP()
        << "DMA-BUF buffers are not supported on this platform; skipping "
           "the test";
  }

  LrtTensorBuffer buffer;
  ASSERT_EQ(
      LrtCreateManagedTensorBuffer(kLrtTensorBufferTypeDmaBuf, kTensorType,
                                   sizeof(kTensorData), &buffer),
      kLrtStatusOk);

  LrtTensorBufferType buffer_type;
  ASSERT_EQ(LrtGetTensorBufferType(buffer, &buffer_type), kLrtStatusOk);
  ASSERT_EQ(buffer_type, kLrtTensorBufferTypeDmaBuf);

  LrtRankedTensorType tensor_type;
  ASSERT_EQ(LrtGetTensorBufferTensorType(buffer, &tensor_type), kLrtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLrtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_NE(tensor_type.layout.strides, nullptr);

  size_t size;
  ASSERT_EQ(LrtGetTensorBufferSize(buffer, &size), kLrtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LrtGetTensorBufferOffset(buffer, &offset), kLrtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  LrtDestroyTensorBuffer(buffer);
}
#endif  // LRT_HAS_DMABUF_SUPPORT

#if LRT_HAS_FASTRPC_SUPPORT
TEST(TensorBuffer, FastRpc) {
  if (!lrt::internal::FastRpcBuffer::IsSupported()) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }

  LrtTensorBuffer buffer;
  ASSERT_EQ(
      LrtCreateManagedTensorBuffer(kLrtTensorBufferTypeFastRpc, kTensorType,
                                   sizeof(kTensorData), &buffer),
      kLrtStatusOk);

  LrtTensorBufferType buffer_type;
  ASSERT_EQ(LrtGetTensorBufferType(buffer, &buffer_type), kLrtStatusOk);
  ASSERT_EQ(buffer_type, kLrtTensorBufferTypeFastRpc);

  LrtRankedTensorType tensor_type;
  ASSERT_EQ(LrtGetTensorBufferTensorType(buffer, &tensor_type), kLrtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLrtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_TRUE(!tensor_type.layout.strides);

  size_t size;
  ASSERT_EQ(LrtGetTensorBufferSize(buffer, &size), kLrtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LrtGetTensorBufferOffset(buffer, &offset), kLrtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  ASSERT_EQ(LrtLockTensorBuffer(buffer, &host_mem_addr, /*event=*/nullptr),
            kLrtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LrtUnlockTensorBuffer(buffer), kLrtStatusOk);

  LrtDestroyTensorBuffer(buffer);
}
#endif  // LRT_HAS_FASTRPC_SUPPORT
