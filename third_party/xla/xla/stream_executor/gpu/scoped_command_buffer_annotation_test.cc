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

#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor {
namespace {

TEST(ScopedCommandBufferAnnotationTest, EmptyStack) {
  EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "");
}

TEST(ScopedCommandBufferAnnotationTest, SingleScope) {
  {
    ScopedCommandBufferAnnotation annotation("layer1");
    EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "layer1");
  }
  EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "");
}

TEST(ScopedCommandBufferAnnotationTest, NestedScopes) {
  {
    ScopedCommandBufferAnnotation annotation1("outer");
    EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "outer");
    {
      ScopedCommandBufferAnnotation annotation2("inner");
      EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "inner");
    }
    EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "outer");
  }
  EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "");
}

TEST(ScopedCommandBufferAnnotationTest, MultiThreadedIndependence) {
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

    pool.Schedule([] {
      ScopedCommandBufferAnnotation annotation("thread1_annotation");
      EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(),
                "thread1_annotation");
    });

    pool.Schedule([] {
      ScopedCommandBufferAnnotation annotation("thread2_annotation");
      EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(),
                "thread2_annotation");
    });
  }
  // Pool destruction waits for workers to finish.
  EXPECT_EQ(ScopedCommandBufferAnnotation::GetCurrentAnnotation(), "");
}

}  // namespace
}  // namespace stream_executor
