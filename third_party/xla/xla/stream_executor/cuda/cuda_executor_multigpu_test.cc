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
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_init.h"
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
absl::StatusOr<stream_executor::DeviceMemoryBase> AllocateInitializedMemory(
    CudaExecutor* executor, size_t size, T value) {
  size_t num_elements = size / sizeof(T);
  stream_executor::DeviceMemoryBase device_memory = executor->Allocate(
      size, static_cast<int64_t>(stream_executor::MemoryType::kP2P));
  if (device_memory.opaque() == nullptr) {
    return absl::InternalError("Failed to allocate memory.");
  }
  std::vector<T> device_memory_vector(num_elements, value);

  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(
      &device_memory, device_memory_vector.data(), size));
  return device_memory;
}

template <typename T>
absl::Status CheckMemory(CudaExecutor* executor,
                         stream_executor::DeviceMemoryBase device_memory,
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

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryResubscriptionFails) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  std::unique_ptr<CudaExecutor::MulticastMemory> multicast_memory;
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
  std::unique_ptr<CudaExecutor::MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory,
                          executors[0]->CreateMulticastMemory(1024, 2));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(
      multicast_memory->MapMemory(reinterpret_cast<void*>(1), executors[0]),
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
  std::unique_ptr<CudaExecutor::MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory,
                          executors[0]->CreateMulticastMemory(1024, 2));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "All devices are already subscribed."));
  ;
}

TEST(CudaExecutorMultiGpuTest, CudaMulticastMemoryUsingNonVmmMemory) {
  std::vector<CudaExecutor*> executors = {
      static_cast<CudaExecutor*>(GetGpuExecutor(0)),
      static_cast<CudaExecutor*>(GetGpuExecutor(1))};
  if (!executors[0]->is_multicast_supported()) {
    GTEST_SKIP() << "Test requires multicast support.";
  }
  const int64_t kNumDevices = 2;
  std::unique_ptr<CudaExecutor::MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(
      multicast_memory, executors[0]->CreateMulticastMemory(1024, kNumDevices));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());

  DeviceMemoryBase device_memory = executors[0]->Allocate(8, 0);
  EXPECT_THAT(
      multicast_memory->MapMemory(device_memory.opaque(), executors[0]),
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
  std::unique_ptr<CudaExecutor::MulticastMemory> multicast_memory;
  TF_ASSERT_OK_AND_ASSIGN(multicast_memory, executors[0]->CreateMulticastMemory(
                                                kMemorySize, kNumDevices));
  EXPECT_THAT(multicast_memory->SubscribeDevice(0), IsOk());
  EXPECT_THAT(multicast_memory->SubscribeDevice(1), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceMemoryBase first_device_memory,
      AllocateInitializedMemory(executors[0], kMemorySize, kValue));

  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceMemoryBase output_device_memory,
      AllocateInitializedMemory(executors[0], kMemorySize, 0));
  TF_ASSERT_OK_AND_ASSIGN(
      void* first_device_multicast_ptr,
      multicast_memory->MapMemory(first_device_memory.opaque(), executors[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceMemoryBase second_device_memory,
      AllocateInitializedMemory(executors[1], kMemorySize, kValue));
  EXPECT_THAT(
      multicast_memory->MapMemory(second_device_memory.opaque(), executors[1]),
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
