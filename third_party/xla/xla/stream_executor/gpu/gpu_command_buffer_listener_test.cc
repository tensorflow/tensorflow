/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_command_buffer_listener.h"

#include <cstddef>
#include <thread>  // NOLINT(build/c++11)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_command_buffer_listener_test_helper.h"

namespace stream_executor::gpu {
namespace {

class MockGpuCommandBufferListener : public GpuCommandBufferListener {
 public:
  MOCK_METHOD(void, OnRegisterNodeAnnotation,
              (void* graph, GpuCommandBuffer::GraphNodeHandle node,
               absl::string_view annotation),
              (override));
  MOCK_METHOD(void, OnRegisterGraphSize, (void* graph, size_t size),
              (override));
  MOCK_METHOD(void, OnRegisterChildGraph,
              (void* parent_graph, void* child_graph,
               GpuCommandBuffer::GraphNodeHandle child_node,
               bool is_conditional),
              (override));
  MOCK_METHOD(void, OnRegisterGraphExec, (void* graph_exec, void* graph),
              (override));
  MOCK_METHOD(void, OnUnregisterGraphExec, (void* graph_exec), (override));
  MOCK_METHOD(void, OnUnregisterGraphAnnotations, (void* graph), (override));
};

TEST(GpuCommandBufferListenerTest, RegistrationAndRetrieval) {
  EXPECT_EQ(GetGpuCommandBufferListener(), nullptr);

  MockGpuCommandBufferListener listener1;
  MockGpuCommandBufferListener listener2;

  // First registration succeeds.
  EXPECT_TRUE(RegisterGpuCommandBufferListener(&listener1));
  EXPECT_EQ(GetGpuCommandBufferListener(), &listener1);

  // Overwriting with another listener fails.
  EXPECT_FALSE(RegisterGpuCommandBufferListener(&listener2));
  EXPECT_EQ(GetGpuCommandBufferListener(), &listener1);

  // Unregistering with wrong listener pointer fails.
  EXPECT_FALSE(UnregisterGpuCommandBufferListener(&listener2));
  EXPECT_EQ(GetGpuCommandBufferListener(), &listener1);

  // Unregistering with correct pointer succeeds.
  EXPECT_TRUE(UnregisterGpuCommandBufferListener(&listener1));
  EXPECT_EQ(GetGpuCommandBufferListener(), nullptr);

  // Null pointers are rejected.
  EXPECT_FALSE(RegisterGpuCommandBufferListener(nullptr));
  EXPECT_FALSE(UnregisterGpuCommandBufferListener(nullptr));
}

TEST(GpuCommandBufferListenerTest, ThreadLocalOverride) {
  EXPECT_EQ(GetGpuCommandBufferListener(), nullptr);

  MockGpuCommandBufferListener global_listener;
  MockGpuCommandBufferListener override_listener;

  EXPECT_TRUE(RegisterGpuCommandBufferListener(&global_listener));
  EXPECT_EQ(GetGpuCommandBufferListener(), &global_listener);

  {
    ScopedGpuCommandBufferListenerOverrideForTesting listener_override(
        &override_listener);
    EXPECT_EQ(GetGpuCommandBufferListener(), &override_listener);

    // Verify it is thread local.
    std::thread t([&global_listener]() {
      // Other thread still sees the global listener.
      EXPECT_EQ(GetGpuCommandBufferListener(), &global_listener);
    });
    t.join();
  }

  // Restored to global after scope exits.
  EXPECT_EQ(GetGpuCommandBufferListener(), &global_listener);

  EXPECT_TRUE(UnregisterGpuCommandBufferListener(&global_listener));
}

}  // namespace
}  // namespace stream_executor::gpu
