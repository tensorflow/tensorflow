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
#include "absl/status/status_matchers.h"
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

}  // namespace
}  // namespace xla
