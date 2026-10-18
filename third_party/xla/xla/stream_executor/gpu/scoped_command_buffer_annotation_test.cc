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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor {
namespace {

using Scope = ScopedCommandBufferAnnotation;

TEST(ScopedCommandBufferAnnotationTest, EnabledRootRecordsNestedPushes) {
  EXPECT_FALSE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
  {
    auto root = Scope::Root(/*enabled=*/true, "XlaModule:#m#");
    EXPECT_TRUE(Scope::IsActive());
    EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#m#");
    {
      Scope inner("Thunk:#hlo_op=fusion.1#");
      EXPECT_EQ(Scope::GetCurrentAnnotation(),
                "XlaModule:#m#::Thunk:#hlo_op=fusion.1#");
      {
        // An empty push adds nothing.
        Scope empty("");
        EXPECT_EQ(Scope::GetCurrentAnnotation(),
                  "XlaModule:#m#::Thunk:#hlo_op=fusion.1#");
      }
    }
    EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#m#");
  }
  EXPECT_FALSE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
}

TEST(ScopedCommandBufferAnnotationTest, DisabledRootDoesNothing) {
  auto root = Scope::Root(/*enabled=*/false, "XlaModule:#m#");
  EXPECT_FALSE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
  Scope inner("Thunk:#hlo_op=fusion.1#");
  EXPECT_FALSE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
}

TEST(ScopedCommandBufferAnnotationTest, PushWithoutRootDoesNothing) {
  Scope eager("XlaModule:#m#");
  EXPECT_FALSE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());

  int calls = 0;
  Scope lazy([&] {
    ++calls;
    return std::string("Thunk:#hlo_op=fusion.1#");
  });
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
}

TEST(ScopedCommandBufferAnnotationTest, LazyGeneratorCalledOnceInsideRoot) {
  auto root = Scope::Root(/*enabled=*/true, "XlaModule:#m#");
  int string_calls = 0;
  int view_calls = 0;
  {
    Scope from_string([&] {
      ++string_calls;
      return std::string("Thunk:#hlo_op=fusion.1#");
    });
    EXPECT_EQ(string_calls, 1);
    EXPECT_EQ(Scope::GetCurrentAnnotation(),
              "XlaModule:#m#::Thunk:#hlo_op=fusion.1#");
    {
      Scope from_view([&] {
        ++view_calls;
        return absl::string_view("Thunk:#hlo_op=fusion.2#");
      });
      EXPECT_EQ(view_calls, 1);
      EXPECT_EQ(Scope::GetCurrentAnnotation(),
                "XlaModule:#m#::Thunk:#hlo_op=fusion.1#::"
                "Thunk:#hlo_op=fusion.2#");
    }
  }
  EXPECT_EQ(string_calls, 1);
  EXPECT_EQ(view_calls, 1);
  EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#m#");
}

TEST(ScopedCommandBufferAnnotationTest, NestedRootsKeepRecordingOn) {
  {
    auto outer = Scope::Root(/*enabled=*/true, "outer");
    {
      auto inner = Scope::Root(/*enabled=*/true, "inner");
      EXPECT_TRUE(Scope::IsActive());
      EXPECT_EQ(Scope::GetCurrentAnnotation(), "outer::inner");
    }
    // The outer root keeps recording on after the inner one closes.
    EXPECT_TRUE(Scope::IsActive());
    EXPECT_EQ(Scope::GetCurrentAnnotation(), "outer");
    {
      // A disabled inner root does not turn recording off.
      auto disabled = Scope::Root(/*enabled=*/false, "disabled");
      EXPECT_TRUE(Scope::IsActive());
      EXPECT_EQ(Scope::GetCurrentAnnotation(), "outer");
    }
  }
  EXPECT_FALSE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
}

TEST(ScopedCommandBufferAnnotationTest, EmptyRootEnablesRecording) {
  auto root = Scope::Root(/*enabled=*/true, "");
  EXPECT_TRUE(Scope::IsActive());
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
  {
    // No leading "::" after an empty root.
    Scope inner("Thunk:#hlo_op=fusion.1#");
    EXPECT_EQ(Scope::GetCurrentAnnotation(), "Thunk:#hlo_op=fusion.1#");
  }
  EXPECT_TRUE(Scope::GetCurrentAnnotation().empty());
}

TEST(ScopedCommandBufferAnnotationTest, StringArgumentsUseEagerConstructor) {
  // `const char*` and `std::string` are not invocable, so they bind to the
  // `absl::string_view` constructor rather than the lazy one.
  auto root = Scope::Root(/*enabled=*/true, "XlaModule:#m#");
  {
    Scope literal("literal");
    EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#m#::literal");
  }
  {
    const std::string annotation = "string";
    Scope from_string(annotation);
    EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#m#::string");
  }
  EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#m#");
}

TEST(ScopedCommandBufferAnnotationTest, ThreadIsolation) {
  auto main_root = Scope::Root(/*enabled=*/true, "XlaModule:#module=main#");
  EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#module=main#");

  bool active_on_other_thread = true;
  bool current_empty_on_other_thread = false;
  std::unique_ptr<tsl::Thread> other(
      tsl::Env::Default()->StartThread(tsl::ThreadOptions(), "other", [&] {
        active_on_other_thread = Scope::IsActive();
        current_empty_on_other_thread = Scope::GetCurrentAnnotation().empty();
      }));
  other.reset();

  EXPECT_FALSE(active_on_other_thread);
  EXPECT_TRUE(current_empty_on_other_thread);

  EXPECT_TRUE(Scope::IsActive());
  EXPECT_EQ(Scope::GetCurrentAnnotation(), "XlaModule:#module=main#");
}

}  // namespace
}  // namespace stream_executor
