/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

#include <string>

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

TEST(ScopedMemoryDebugAnnotationTest, TlsDestructionOrder) {
  // We want to ensure that accessing CurrentAnnotation() from a thread_local
  // destructor does not crash if the ThreadMemoryDebugAnnotation is destructed
  // before the test's thread_local object.
  struct DestructorTester {
    ~DestructorTester() {
      // This will invoke ScopedMemoryDebugAnnotation constructor during thread
      // exit. If ThreadMemoryDebugAnnotation() was already destructed and is
      // not using absl::NoDestructor, this would cause a use-after-destruction
      // crash.
      ScopedMemoryDebugAnnotation anno("test_op");
    }
  };

  // Run the logic in a separate thread so its thread_local variables
  // are destructed when the thread joins.
  std::unique_ptr<tsl::Thread> t(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "TestThread", []() {
        // 1. Construct the test object first, so it is destructed last.
        static thread_local DestructorTester tester;

        // 2. Initialize the thread-local MemoryDebugAnnotation so it is
        // constructed
        //    second, and therefore destructed first during thread exit.
        // Make sure we pass a lambda that captures something to force the
        // std::function to allocate heap memory, making use-after-free highly
        // likely to crash without the fix.
        std::string large_string(1000, 'x');
        ScopedMemoryDebugAnnotation init(
            "init_op", "region", 0, [large_string]() { return large_string; });
      }));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
