/* Copyright 2022 The OpenXLA Authors.

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

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/notification.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::tsl::BlockUntilReady;
using ::tsl::MakeConstructedAsyncValueRef;
using ::tsl::thread::ThreadPool;

TEST(TrackedCpuDeviceBufferTest, Basic) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  PjRtMemorySpace* memory_space = client->memory_spaces()[0];
  std::string expected = "tracked_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, CpuRawBuffer::Allocate(memory_space, expected.size()));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer->buffer()->untyped_data(), expected.data(),
                expected.size());
    definition_event.SetStateConcrete();
  });

  absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events;
  definition_events.push_back(PjRtDeviceEventRef(definition_event));
  AbstractTrackedDeviceBuffer tracked_buffer(
      buffer, std::move(definition_events), true);

  ABSL_ASSERT_OK(tracked_buffer.BlockForOperationsToComplete(memory_space));

  auto result =
      absl::down_cast<CpuRawBuffer*>(tracked_buffer.raw_buffer().get())
          ->buffer();
  ASSERT_TRUE(result.IsAvailable());
  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

TEST(TrackedCpuDeviceBufferTest, BasicError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  PjRtMemorySpace* memory_space = client->memory_spaces()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          CpuRawBuffer::Allocate(memory_space, 64));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    definition_event.SetError(
        Internal("tracked_cpu_device_buffer_test error."));
  });

  absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events;
  definition_events.push_back(PjRtDeviceEventRef(definition_event));
  AbstractTrackedDeviceBuffer tracked_buffer(
      buffer, std::move(definition_events), true);

  EXPECT_FALSE(tracked_buffer.BlockForOperationsToComplete(memory_space).ok());

  ASSERT_TRUE(definition_event.IsError());
  EXPECT_EQ(definition_event.GetError().message(),
            "tracked_cpu_device_buffer_test error.");
}

TEST(TrackedCpuDeviceBufferTest, DelayedAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  PjRtMemorySpace* memory_space = client->memory_spaces()[0];
  std::string expected = "tracked_cpu_device_buffer_test";

  auto buffer = CpuDeviceMemory::CreateDelayedMemory();
  auto malloc_event = MakeConstructedAsyncValueRef<CpuEvent>();
  malloc_event.AndThen([buffer, buffer_size = expected.size()]() mutable {
    CHECK_OK(CpuDeviceMemory::AllocateInto(buffer_size, buffer.AsPtr()));
  });

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();
  absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events;
  definition_events.push_back(PjRtDeviceEventRef(definition_event));
  AbstractTrackedDeviceBuffer tracked_buffer(
      tsl::MakeRef<CpuRawBuffer>(memory_space, buffer, expected.size(),
                                 /*is_mutable=*/true),
      std::move(definition_events), true);

  auto result =
      absl::down_cast<CpuRawBuffer*>(tracked_buffer.raw_buffer().get())
          ->buffer();
  ASSERT_FALSE(result.IsAvailable());
  ASSERT_EQ(tracked_buffer.raw_buffer()->GetOnDeviceSizeInBytes(),
            expected.size());

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    malloc_event.SetStateConcrete();
    std::memcpy(buffer->untyped_data(), expected.data(), expected.size());
    definition_event.SetStateConcrete();
  });

  ABSL_ASSERT_OK(tracked_buffer.BlockForOperationsToComplete(memory_space));

  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

TEST(TrackedCpuDeviceBufferTest, SlicedMemoryParentErrorSecureAborts) {
  // We must perform the orchestration entirely inside the EXPECT_DEATH block.
  // This is because EXPECT_DEATH forks, and we want to perform the
  // multi-threaded synchronization cleanly inside the child process.
  EXPECT_DEATH(
      {
        // 1. Create an unconstructed parent buffer placeholder
        tsl::AsyncValueRef<CpuDeviceMemory> parent =
            CpuDeviceMemory::CreateDelayedMemory();

        // 2. Create a sliced buffer from the unconstructed parent.
        // This internally calls parent.AndThen(...) which adds the slice's
        // transition waiter (W2).
        tsl::AsyncValueRef<CpuDeviceMemory> slice =
            CpuDeviceMemory::CreateSlicedMemory(parent.CopyRef(), /*offset=*/10,
                                                /*size=*/20);

        // 3. Retrieve the slice payload reference while the slice is in
        // constructed state. Note: slice.get() accesses the payload
        // CpuDeviceMemorySlice object.
        CpuDeviceMemory& slice_payload = slice.get();

        // 4. Setup synchronization notifications.
        absl::Notification w1_has_started;
        absl::Notification w1_can_proceed;

        // 5. Add a blocker waiter (W1) to parent.
        // Because AndThen prepends to the LIFO waiter list, W1 is added after
        // W2 and thus will be executed BEFORE W2 when parent becomes available.
        parent.AndThen([&]() {
          w1_has_started.Notify();
          w1_can_proceed.WaitForNotification();
        });

        // 6. Spawn a helper thread to set the parent error.
        // We must do this on a background thread because parent.SetError() will
        // block synchronously on the current thread when executing W1.
        std::unique_ptr<tsl::Thread> setter_thread(
            tsl::Env::Default()->StartThread(
                tsl::ThreadOptions(), "setter_thread", [&]() {
                  parent.SetError(absl::ResourceExhaustedError(
                      "Out of Memory allocating parent buffer"));
                }));

        // 7. Wait until W1 has started executing.
        // At this point, parent has transitioned to kError state, but the slice
        // transition waiter (W2) has NOT run yet because the waiter loop is
        // blocked in W1.
        w1_has_started.WaitForNotification();

        // 8. Attempt to call untyped_data on the slice.
        // Under the secure remediation, because parent is in kError, this must
        // trigger a clean LOG(FATAL) containing the parent's error status.
        // Since W2 hasn't run yet, the slice is still in kConstructed state and
        // has not been destructed, so slice_payload points to a fully valid,
        // non-type-confused object.
        void* ptr = slice_payload.untyped_data();
        (void)ptr;

        // Clean up (in case EXPECT_DEATH didn't kill the process, though it
        // should have). Deleting the tsl::Thread pointer will block and join
        // it.
        w1_can_proceed.Notify();
        setter_thread.reset();
      },
      "Accessing untyped_data on a sliced buffer whose parent failed: "
      "RESOURCE_EXHAUSTED: Out of Memory allocating parent buffer");
}

}  // namespace
}  // namespace xla
