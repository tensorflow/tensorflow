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

#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
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
  std::string expected = "tracked_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          CpuDeviceMemory::Allocate(expected.size()));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer->untyped_data(), expected.data(), expected.size());
    definition_event.SetStateConcrete();
  });

  TrackedCpuDeviceBuffer tracked_buffer(
      /*owns_buffers=*/true, buffer, definition_event);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  auto result = tracked_buffer.buffer();
  ASSERT_TRUE(result.IsAvailable());
  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

TEST(TrackedCpuDeviceBufferTest, BasicError) {
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, CpuDeviceMemory::Allocate(64));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    definition_event.SetError(
        Internal("tracked_cpu_device_buffer_test error."));
  });

  TrackedCpuDeviceBuffer tracked_buffer(
      /*owns_buffers=*/true, buffer, definition_event);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  ASSERT_TRUE(tracked_buffer.definition_event().IsError());
  EXPECT_EQ(tracked_buffer.definition_event().GetError().message(),
            "tracked_cpu_device_buffer_test error.");
}

TEST(TrackedCpuDeviceBufferTest, DelayedAllocation) {
  std::string expected = "tracked_cpu_device_buffer_test";

  auto buffer = CpuDeviceMemory::CreateDelayedMemory();
  auto malloc_event = MakeConstructedAsyncValueRef<CpuEvent>();
  malloc_event.AndThen([buffer, buffer_size = expected.size()]() mutable {
    CHECK_OK(CpuDeviceMemory::AllocateInto(buffer_size, buffer.AsPtr()));
  });

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();
  TrackedCpuDeviceBuffer tracked_buffer(/*owns_buffers=*/true, buffer,
                                        expected.size(), definition_event);
  auto result = tracked_buffer.buffer();
  ASSERT_FALSE(result.IsAvailable());
  ASSERT_EQ(tracked_buffer.BufferSize(), expected.size());

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    malloc_event.SetStateConcrete();
    std::memcpy(buffer->untyped_data(), expected.data(), expected.size());
    definition_event.SetStateConcrete();
  });

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

}  // namespace
}  // namespace xla
