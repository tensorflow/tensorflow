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

#include "xla/backends/gpu/runtime/while_loop.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace xla::gpu {

TEST(WhileLoopTest, NotInsideLoop) { EXPECT_EQ(IsInsideWhileLoop(), nullptr); }

TEST(WhileLoopTest, EnterExit) {
  {
    ScopedWhileLoop loop("while.0");
    EXPECT_EQ(loop.loop_name(), "while.0");
    EXPECT_EQ(loop.trip_count(), std::nullopt);
    EXPECT_EQ(loop.loop_depth(), 0);
    EXPECT_EQ(loop.loop_iteration(), 0);
    EXPECT_NE(IsInsideWhileLoop(), nullptr);
  }
  EXPECT_EQ(IsInsideWhileLoop(), nullptr);
}

TEST(WhileLoopTest, Increment) {
  ScopedWhileLoop loop("while.0", 10);
  EXPECT_EQ(loop.trip_count(), 10);
  EXPECT_EQ(loop.loop_iteration(), 0);

  loop.IncLoopIteration();
  EXPECT_EQ(loop.loop_iteration(), 1);

  loop.IncLoopIteration();
  EXPECT_EQ(loop.loop_iteration(), 2);
}

TEST(WhileLoopTest, NestedLoops) {
  ScopedWhileLoop outer("while.0");
  EXPECT_EQ(outer.loop_depth(), 0);

  {
    ScopedWhileLoop inner("while.1");
    EXPECT_EQ(inner.loop_depth(), 1);
    EXPECT_EQ(IsInsideWhileLoop()->loop_name, "while.1");

    inner.IncLoopIteration();
    EXPECT_EQ(inner.loop_iteration(), 1);
    EXPECT_EQ(outer.loop_iteration(), 0);
  }

  EXPECT_EQ(IsInsideWhileLoop()->loop_name, "while.0");
  EXPECT_EQ(outer.loop_iteration(), 0);
}

TEST(WhileLoopTest, PointerStability) {
  std::vector<std::unique_ptr<ScopedWhileLoop>> loops(10);
  for (int i = 0; i < 10; ++i) {
    loops[i] = std::make_unique<ScopedWhileLoop>(absl::StrCat("while.", i));
  }

  // All previously created loops remain accessible after 10 nested pushes.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(loops[i]->loop_name(), absl::StrCat("while.", i));
    EXPECT_EQ(loops[i]->loop_depth(), i);
    EXPECT_EQ(loops[i]->loop_iteration(), 0);
  }

  for (int i = 9; i >= 0; --i) {
    EXPECT_EQ(IsInsideWhileLoop()->loop_name, absl::StrCat("while.", i));
    loops.pop_back();
  }
  EXPECT_EQ(IsInsideWhileLoop(), nullptr);
}

}  // namespace xla::gpu
