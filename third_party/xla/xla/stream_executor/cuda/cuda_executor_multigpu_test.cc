/* Copyright 2025 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_executor_multigpu_test_kernels.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::NotNull;

template <typename T>
absl::StatusOr<stream_executor::DeviceAddressBase> AllocateInitializedMemory(
    CudaExecutor* executor, size_t size, size_t offset, T value) {
  stream_executor::DeviceAddressBase device_memory = executor->Allocate(
      size + offset, static_cast<int64_t>(stream_executor::MemorySpace::kP2P));
  if (device_memory.opaque() == nullptr) {
    return absl::InternalError("Failed to allocate memory.");
  }

  size_t num_initialized_elements = size / sizeof(T);
  std::vector<T> device_memory_vector(num_initialized_elements, value);

  auto stride_memory = device_memory.GetByteSlice(offset, size);
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(
      &stride_memory, device_memory_vector.data(), size));
  return stride_memory;
}

template <typename T>
absl::Status CheckMemory(CudaExecutor* executor,
                         stream_executor::DeviceAddressBase device_memory,
                         T expected_value) {
  size_t num_elements = device_memory.size() / sizeof(T);
  std::vector<T> device_memory_vector(num_elements, 0);
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(
      device_memory_vector.data(), device_memory, device_memory.size()));
  for (int i = 0; i < device_memory_vector.size(); ++i) {
    EXPECT_EQ(device_memory_vector[i], expected_value);
  }

  return absl::OkStatus();
}

StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      PlatformManager::PlatformWithName(stream_executor::GpuPlatformName())
          .value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

TEST(CudaExecutorMultiGpuTest, PeerAccess) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};

  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  EXPECT_TRUE(executors[0]->CanEnablePeerAccessTo(0));
  EXPECT_TRUE(executors[0]->CanEnablePeerAccessTo(1));
  EXPECT_TRUE(executors[1]->CanEnablePeerAccessTo(0));
  EXPECT_TRUE(executors[1]->CanEnablePeerAccessTo(1));
  EXPECT_FALSE(executors[0]->CanEnablePeerAccessTo(100));
}

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryResubscriptionFails) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory,
                          executors[0]->CreateMulticastMemory(1024, 2));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(0),
              StatusIs(absl::StatusCode::kInternal,
                       "CUDA error: : CUDA_ERROR_UNKNOWN: unknown error"));
}

TEST(CudaExecutorMultiGpuTest, AllDevicesMustBeSubscribedBeforeMapping) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory,
                          executors[0]->CreateMulticastMemory(1024, 2));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  DeviceAddressBase device_memory(reinterpret_cast<void*>(1), 1);
  EXPECT_THAT(multicast_memory->MapMemory(device_memory, executors[0]),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "All devices should be subscribed."));
  ;
}

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemorySubscribeMoreDevices) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory,
                          executors[0]->CreateMulticastMemory(1024, 2));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "All devices are already subscribed."));
}

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryUsingNonVmmMemory) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  const int64_t kNumDevices = 2;
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(
      multicast_memory, executors[0]->CreateMulticastMemory(1024, kNumDevices));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());

  DeviceAddressBase device_memory = executors[0]->Allocate(8, 0);
  EXPECT_THAT(
      multicast_memory->MapMemory(device_memory, executors[0]),
      StatusIs(absl::StatusCode::kInternal,
               "CUDA error: : CUDA_ERROR_INVALID_VALUE: invalid argument"));
}

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryUsingVmmMemory) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  const int kNumDevices = 2;
  const int kNumElements = 8;
  const size_t kMemorySize = kNumElements * sizeof(int);
  const int kValue = 2;
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory, executors[0]->CreateMulticastMemory(
                                                kMemorySize, kNumDevices));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase first_device_memory,
      AllocateInitializedMemory(executors[0], kMemorySize, 0, kValue));

  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase output_device_memory,
      AllocateInitializedMemory(executors[0], kMemorySize, 0, 0));
  TF_ASSERT_OK_AND_ASSIGN(
      void* first_device_multicast_ptr,
      multicast_memory->MapMemory(first_device_memory, executors[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase second_device_memory,
      AllocateInitializedMemory(executors[1], kMemorySize, 0, kValue));
  EXPECT_THAT(multicast_memory->MapMemory(second_device_memory, executors[1]),
              IsOkAndHolds(NotNull()));

  EXPECT_THAT(
      MulticastReduce((int*)first_device_multicast_ptr,
                      (int*)output_device_memory.opaque(), kNumElements),
      IsOk());

  const int kExpectedValue = kValue * kNumDevices;
  EXPECT_THAT(CheckMemory(executors[0], output_device_memory, kExpectedValue),
              IsOk());
}

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryMapDifferentSlicesUnaligned) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  const int64_t kNumDevices = 2;
  const int64_t kNumElements = 8;
  const int64_t kMappedMemorySize = kNumElements * sizeof(int);
  const int kValue = 2;
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(
      multicast_memory,
      executors[0]->CreateMulticastMemory(kMappedMemorySize, kNumDevices));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(size_t vmm_granularity,
                          executors[0]->GetVmmGranularity());
  // Allocate memory with unaligned offset.
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase first_device_mapped_memory,
      AllocateInitializedMemory(
          executors[0],
          // Add granularity to make sure that there is
          // enough memory after adding offset to map with multicast object.
          kMappedMemorySize + vmm_granularity, kMappedMemorySize, kValue));
  EXPECT_THAT(
      multicast_memory->MapMemory(first_device_mapped_memory, executors[0]),
      StatusIs(absl::StatusCode::kInternal,
               "CUDA error: : CUDA_ERROR_INVALID_VALUE: invalid argument"));
}

// Slices mapping works only when offset is aligned with the VMM granularity.
TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryMapDifferentSlices) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  const int64_t kNumDevices = 2;
  const int64_t kNumElements = 8;
  const int64_t kMappedMemorySize = kNumElements * sizeof(int);
  const int kValue = 2;
  std::unique_ptr<MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(
      multicast_memory,
      executors[0]->CreateMulticastMemory(kMappedMemorySize, kNumDevices));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(size_t vmm_granularity,
                          executors[0]->GetVmmGranularity());
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase first_device_mapped_memory,
      AllocateInitializedMemory(executors[0], kMappedMemorySize,
                                vmm_granularity, kValue));
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase output_device_memory,
      AllocateInitializedMemory(executors[0], kMappedMemorySize, 0, 0));
  TF_ASSERT_OK_AND_ASSIGN(
      void* first_device_multicast_ptr,
      multicast_memory->MapMemory(first_device_mapped_memory, executors[0]));

  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceAddressBase second_device_mapped_memory,
      AllocateInitializedMemory(executors[1], kMappedMemorySize, 0, kValue));
  EXPECT_THAT(
      multicast_memory->MapMemory(second_device_mapped_memory, executors[1]),
      IsOkAndHolds(NotNull()));

  EXPECT_THAT(
      MulticastReduce((int*)first_device_multicast_ptr,
                      (int*)output_device_memory.opaque(), kNumElements),
      IsOk());

  const int kExpectedValue = kValue * kNumDevices;
  EXPECT_THAT(CheckMemory(executors[0], output_device_memory, kExpectedValue),
              IsOk());
}
}  // namespace
}  // namespace stream_executor::gpu
