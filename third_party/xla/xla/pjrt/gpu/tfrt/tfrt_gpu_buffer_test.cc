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

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace {

using ::tsl::thread::ThreadPool;

TEST(TfrtGpuBufferTest, CreateBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(), size_in_bytes));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      tsl::MakeAvailableAsyncValueRef<GpuEvent>());
  auto memory_space = device->default_memory_space().value();
  auto buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  EXPECT_EQ(buffer->on_device_shape(), on_device_shape);
  EXPECT_EQ(buffer->device(), device);
  EXPECT_EQ(buffer->client(), client.get());
  EXPECT_EQ(buffer->memory_space(), memory_space);
  EXPECT_EQ(buffer->GetOnDeviceSizeInBytes().value(), size_in_bytes);
}

TEST(TfrtGpuBufferTest, AcquireExternalReference) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(), size_in_bytes));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));
  auto definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref), definition_event);
  auto memory_space = device->default_memory_space().value();
  auto buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  ThreadPool thread_pool(tsl::Env::Default(), "gpu_buffer_test",
                         /*num_threads=*/4);

  absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>> ref_status;
  auto ref_acquired_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  thread_pool.Schedule([&]() {
    ref_status = buffer->AcquireExternalReference();
    ref_acquired_event.SetStateConcrete();
  });
  // AcquireExternalReference should block until the definition event is
  // triggered.
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_FALSE(ref_acquired_event.IsAvailable());

  // Trigger the definition event. AcquireExternalReference should be unblocked.
  definition_event.SetStateConcrete();
  BlockUntilReady(ref_acquired_event.GetAsyncValue());
  EXPECT_OK(ref_status);

  // TODO(b/382117736): external reference should block donation.
}

TEST(TfrtGpuBufferTest, ReleaseDeviceMemoryOwnershipNoWait) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(), size_in_bytes));
  void* device_memory_opaque = device_buffer.buffer().opaque();
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));

  auto definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref), definition_event);

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  std::array usage_events{usage_event.CopyRef()};
  tracked_device_buffer->AddUsageEvents(absl::MakeSpan(usage_events));

  bool destructed = false;
  tracked_device_buffer->deallocation_event().AndThen(
      [&] { destructed = true; });
  auto memory_space = device->default_memory_space().value();
  auto buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  // Release and don't wait for definition or usage events to complete.
  auto ref_status = buffer->ReleaseDeviceMemoryOwnership(
      /*wait_for_operations_to_complete=*/false);
  EXPECT_OK(ref_status);
  auto ref = std::move(ref_status).value();
  EXPECT_EQ(device_memory_opaque, ref->OpaqueDeviceMemoryDataPointer());

  // Release again should return nullptr.
  auto ref_status_2 = buffer->ReleaseDeviceMemoryOwnership(
      /*wait_for_operations_to_complete=*/false);
  EXPECT_OK(ref_status_2);
  EXPECT_EQ(nullptr, ref_status_2.value().get());
}

TEST(TfrtGpuBufferTest, ReleaseDeviceMemoryOwnershipWait) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(), size_in_bytes));
  void* device_memory_opaque = device_buffer.buffer().opaque();
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));

  auto definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref), definition_event);

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  std::array usage_events{usage_event.CopyRef()};
  tracked_device_buffer->AddUsageEvents(absl::MakeSpan(usage_events));

  bool destructed = false;
  tracked_device_buffer->deallocation_event().AndThen(
      [&] { destructed = true; });
  auto memory_space = device->default_memory_space().value();
  auto buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  ThreadPool thread_pool(tsl::Env::Default(), "gpu_buffer_test",
                         /*num_threads=*/4);

  absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>> ref_status;
  auto ref_acquired_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  thread_pool.Schedule([&]() {
    ref_status = buffer->ReleaseDeviceMemoryOwnership(
        /*wait_for_operations_to_complete=*/true);
    ref_acquired_event.SetStateConcrete();
  });
  // AcquireExternalReference should block until the definition event is
  // triggered.
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_FALSE(ref_acquired_event.IsAvailable());

  // Trigger the definition event.
  definition_event.SetStateConcrete();
  EXPECT_FALSE(ref_acquired_event.IsAvailable());

  // Trigger the usage event.
  usage_event.SetStateConcrete();
  BlockUntilReady(ref_acquired_event.GetAsyncValue());
  EXPECT_OK(ref_status);

  // TODO(b/382117736): should also block until donation event is triggered.
  auto ref = std::move(ref_status).value();
  EXPECT_EQ(device_memory_opaque, ref->OpaqueDeviceMemoryDataPointer());

  // Release again should return nullptr.
  auto ref_status_2 = buffer->ReleaseDeviceMemoryOwnership(
      /*wait_for_operations_to_complete=*/false);
  EXPECT_OK(ref_status_2);
  EXPECT_EQ(nullptr, ref_status_2.value().get());
}

TEST(TfrtGpuBufferTest, Delete) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtGpuClient(GpuClientOptions()));

  Shape on_device_shape = ShapeUtil::MakeShapeWithType<int32_t>({4, 4});
  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(client->devices()[0]);
  auto size_in_bytes = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      MaybeOwningGpuMemory::AllocateShared(device->allocator(), size_in_bytes));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));

  auto definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref), definition_event);

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  std::array usage_events{usage_event.CopyRef()};
  tracked_device_buffer->AddUsageEvents(absl::MakeSpan(usage_events));

  bool destructed = false;
  tracked_device_buffer->deallocation_event().AndThen(
      [&] { destructed = true; });
  auto memory_space = device->default_memory_space().value();
  auto buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer),
      tensorflow::down_cast<TfrtGpuClient*>(client.get()), device,
      memory_space);

  // Delete the buffer. The underlying device memory should not be freed until
  // the usage event is triggered.
  buffer->Delete();
  EXPECT_TRUE(buffer->IsDeleted());
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_FALSE(destructed);

  definition_event.SetStateConcrete();
  EXPECT_FALSE(destructed);

  // TODO(b/382117736): should also wait for donation event.

  usage_event.SetStateConcrete();
  EXPECT_TRUE(destructed);
}

}  // namespace
}  // namespace xla
