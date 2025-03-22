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
#include "xla/service/cpu/cpu_event.h"
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
using ::tsl::MakeUnconstructedAsyncValueRef;
using ::tsl::thread::ThreadPool;

TEST(TrackedCpuDeviceBufferTest, Basic) {
  std::string expected = "tracked_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          CpuDeviceMemory::AllocateAvailable(expected.size()));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer->untyped_data(), expected.data(), expected.size());
    definition_event.SetStateConcrete();
  });

  TrackedCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/false, /*owns_buffers=*/true, {buffer}, definition_event,
      /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  auto result = tracked_buffer.Buffers()[0];
  ASSERT_TRUE(result.IsAvailable());
  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

TEST(TrackedCpuDeviceBufferTest, Tuple) {
  std::string expected_0 = "tracked_cpu_device_buffer_test";
  std::string expected_1 = "tuple";
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_0, CpuDeviceMemory::AllocateAvailable(expected_0.size()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_1, CpuDeviceMemory::AllocateAvailable(expected_1.size()));

  auto definition_event_0 = MakeConstructedAsyncValueRef<CpuEvent>();
  auto definition_event_1 = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer_0->untyped_data(), expected_0.data(), expected_0.size());
    definition_event_0.SetStateConcrete();
  });
  thread_pool.Schedule([&]() {
    std::memcpy(buffer_1->untyped_data(), expected_1.data(), expected_1.size());
    definition_event_1.SetStateConcrete();
  });

  TrackedCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/true, /*owns_buffers=*/true, {buffer_0, buffer_1},
      {definition_event_0, definition_event_1},
      /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  auto result_0 = tracked_buffer.Buffers()[0];
  auto result_1 = tracked_buffer.Buffers()[1];
  ASSERT_TRUE(result_0.IsAvailable());
  ASSERT_TRUE(result_1.IsAvailable());
  EXPECT_EQ(std::string(static_cast<const char*>(result_0->untyped_data()),
                        result_0->size_bytes()),
            expected_0);
  EXPECT_EQ(std::string(static_cast<const char*>(result_1->untyped_data()),
                        result_1->size_bytes()),
            expected_1);
}

TEST(TrackedCpuDeviceBufferTest, BasicError) {
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, CpuDeviceMemory::AllocateAvailable(64));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    definition_event.SetError(
        Internal("tracked_cpu_device_buffer_test error."));
  });

  TrackedCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/false, /*owns_buffers=*/true, {buffer}, definition_event,
      /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  ASSERT_TRUE(tracked_buffer.definition_event().IsError());
  EXPECT_EQ(tracked_buffer.definition_event().GetError().message(),
            "tracked_cpu_device_buffer_test error.");
}

TEST(TrackedCpuDeviceBufferTest, TupleError) {
  std::string expected = "tracked_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_0,
                          CpuDeviceMemory::AllocateAvailable(expected.size()));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_1,
                          CpuDeviceMemory::AllocateAvailable(expected.size()));

  auto definition_event_0 = MakeConstructedAsyncValueRef<CpuEvent>();
  auto definition_event_1 = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer_0->untyped_data(), expected.data(), expected.size());
    definition_event_0.SetStateConcrete();
  });
  thread_pool.Schedule([&]() {
    definition_event_1.SetError(
        Internal("tracked_cpu_device_buffer_test tuple error."));
  });

  TrackedCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/true, /*owns_buffers=*/true, {buffer_0, buffer_1},
      {definition_event_0, definition_event_1},
      /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  ASSERT_TRUE(tracked_buffer.definition_event().IsError());
  EXPECT_EQ(tracked_buffer.definition_event().GetError().message(),
            "tracked_cpu_device_buffer_test tuple error.");
}

TEST(TrackedCpuDeviceBufferTest, DelayedAllocation) {
  std::string expected = "tracked_cpu_device_buffer_test";

  auto buffer = MakeUnconstructedAsyncValueRef<CpuDeviceMemory>();
  auto malloc_event = MakeConstructedAsyncValueRef<CpuEvent>();
  malloc_event.AndThen(
      [buffer_copy = buffer.CopyRef(), buffer_size = expected.size()] {
        buffer_copy.emplace(CpuDeviceMemory::Allocate(buffer_size).value());
      });

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();
  TrackedCpuDeviceBuffer tracked_buffer(/*is_tuple=*/false,
                                        /*owns_buffers=*/true, {buffer},
                                        {expected.size()}, definition_event,
                                        /*on_delete_callback_=*/nullptr);
  auto result = tracked_buffer.Buffers()[0];
  ASSERT_FALSE(result.IsAvailable());
  ASSERT_EQ(tracked_buffer.BufferSizes()[0], expected.size());

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

TEST(TrackedCpuDeviceBufferTest, DelayedAllocationTuple) {
  std::string expected_0 = "tracked_cpu_device_buffer_test";
  std::string expected_1 = "tuple";

  auto buffer_0 = MakeUnconstructedAsyncValueRef<CpuDeviceMemory>();
  auto malloc_event_0 = MakeConstructedAsyncValueRef<CpuEvent>();
  malloc_event_0.AndThen(
      [buffer_0_copy = buffer_0.CopyRef(), buffer_0_size = expected_0.size()] {
        buffer_0_copy.emplace(CpuDeviceMemory::Allocate(buffer_0_size).value());
      });
  auto buffer_1 = MakeUnconstructedAsyncValueRef<CpuDeviceMemory>();
  auto malloc_event_1 = MakeConstructedAsyncValueRef<CpuEvent>();
  malloc_event_1.AndThen(
      [buffer_1_copy = buffer_1.CopyRef(), buffer_1_size = expected_1.size()] {
        buffer_1_copy.emplace(CpuDeviceMemory::Allocate(buffer_1_size).value());
      });

  auto definition_event_0 = MakeConstructedAsyncValueRef<CpuEvent>();
  auto definition_event_1 = MakeConstructedAsyncValueRef<CpuEvent>();
  TrackedCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/true,
      /*owns_buffers=*/true, {buffer_0, buffer_1},
      {expected_0.size(), expected_1.size()},
      {definition_event_0, definition_event_1},
      /*on_delete_callback_=*/nullptr);

  auto result_0 = tracked_buffer.Buffers()[0];
  auto result_1 = tracked_buffer.Buffers()[1];
  ASSERT_FALSE(result_0.IsAvailable());
  ASSERT_FALSE(result_1.IsAvailable());
  ASSERT_EQ(tracked_buffer.BufferSizes()[0], expected_0.size());
  ASSERT_EQ(tracked_buffer.BufferSizes()[1], expected_1.size());

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    malloc_event_0.SetStateConcrete();
    std::memcpy(buffer_0->untyped_data(), expected_0.data(), expected_0.size());
    definition_event_0.SetStateConcrete();
  });
  thread_pool.Schedule([&]() {
    malloc_event_1.SetStateConcrete();
    std::memcpy(buffer_1->untyped_data(), expected_1.data(), expected_1.size());
    definition_event_1.SetStateConcrete();
  });

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  EXPECT_EQ(std::string(static_cast<const char*>(result_0->untyped_data()),
                        result_0->size_bytes()),
            expected_0);
  EXPECT_EQ(std::string(static_cast<const char*>(result_1->untyped_data()),
                        result_1->size_bytes()),
            expected_1);
}

}  // namespace
}  // namespace xla
