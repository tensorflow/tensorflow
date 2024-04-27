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

#include "xla/pjrt/cpu/tracked_tfrt_cpu_device_buffer.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

using ::xla::runtime::CpuEvent;

TEST(TrackedTfrtCpuDeviceBufferTest, Basic) {
  std::string expected = "tracked_tfrt_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, MaybeOwningCpuMemory::AllocateShared(expected.size()));

  auto definition_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "tracked_buffer_test",
                                      /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer->data(), expected.data(), expected.size());
    definition_event.SetStateConcrete();
  });

  TrackedTfrtCpuDeviceBuffer tracked_buffer(/*is_tuple=*/false, {buffer},
                                            definition_event,
                                            /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  auto result = tracked_buffer.Buffers()[0];

  EXPECT_EQ(
      std::string(static_cast<const char*>(result->data()), result->size()),
      expected);
}

TEST(TrackedTfrtCpuDeviceBufferTest, Tuple) {
  std::string expected_0 = "tracked_tfrt_cpu_device_buffer_test";
  std::string expected_1 = "tuple";
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_0, MaybeOwningCpuMemory::AllocateShared(expected_0.size()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_1, MaybeOwningCpuMemory::AllocateShared(expected_1.size()));

  auto definition_event_0 = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto definition_event_1 = tsl::MakeConstructedAsyncValueRef<CpuEvent>();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "tracked_buffer_test",
                                      /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer_0->data(), expected_0.data(), expected_0.size());
    definition_event_0.SetStateConcrete();
  });
  thread_pool.Schedule([&]() {
    std::memcpy(buffer_1->data(), expected_1.data(), expected_1.size());
    definition_event_1.SetStateConcrete();
  });

  TrackedTfrtCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/true, {buffer_0, buffer_1},
      {definition_event_0, definition_event_1},
      /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  auto result_0 = tracked_buffer.Buffers()[0];
  auto result_1 = tracked_buffer.Buffers()[1];

  EXPECT_EQ(
      std::string(static_cast<const char*>(result_0->data()), result_0->size()),
      expected_0);
  EXPECT_EQ(
      std::string(static_cast<const char*>(result_1->data()), result_1->size()),
      expected_1);
}

TEST(TrackedTfrtCpuDeviceBufferTest, BasicError) {
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          MaybeOwningCpuMemory::AllocateShared(64));

  auto definition_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "tracked_buffer_test",
                                      /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    definition_event.SetError("tracked_tfrt_cpu_device_buffer_test error.");
  });

  TrackedTfrtCpuDeviceBuffer tracked_buffer(/*is_tuple=*/false, {buffer},
                                            definition_event,
                                            /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  ASSERT_TRUE(tracked_buffer.definition_event().IsError());
  EXPECT_EQ(tracked_buffer.definition_event().GetError().message(),
            "tracked_tfrt_cpu_device_buffer_test error.");
}

TEST(TrackedTfrtCpuDeviceBufferTest, TupleError) {
  std::string expected = "tracked_tfrt_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_0, MaybeOwningCpuMemory::AllocateShared(expected.size()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_1, MaybeOwningCpuMemory::AllocateShared(expected.size()));

  auto definition_event_0 = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto definition_event_1 = tsl::MakeConstructedAsyncValueRef<CpuEvent>();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "tracked_buffer_test",
                                      /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer_0->data(), expected.data(), expected.size());
    definition_event_0.SetStateConcrete();
  });
  thread_pool.Schedule([&]() {
    definition_event_1.SetError(
        "tracked_tfrt_cpu_device_buffer_test tuple error.");
  });

  TrackedTfrtCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/true, {buffer_0, buffer_1},
      {definition_event_0, definition_event_1},
      /*on_delete_callback_=*/nullptr);

  BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());

  ASSERT_TRUE(tracked_buffer.definition_event().IsError());
  EXPECT_EQ(tracked_buffer.definition_event().GetError().message(),
            "tracked_tfrt_cpu_device_buffer_test tuple error.");
}

}  // namespace
}  // namespace xla
