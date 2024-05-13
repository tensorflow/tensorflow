/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/scoped_module_handle.h"

#include <utility>

#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/module_spec.h"
#include "tsl/platform/test.h"

using testing::Return;

namespace stream_executor {
namespace {

TEST(ScopedModuleHandleTest, NoUnloadForNullHandle) {
  ModuleHandle foo;
  MockStreamExecutor executor;
  EXPECT_CALL(executor, UnloadModule).Times(0);
  {
    ScopedModuleHandle first(&executor, foo);
    ScopedModuleHandle second = std::move(first);
    ScopedModuleHandle third(&executor, foo);
    third = std::move(second);
  }
}

TEST(ScopedModuleHandleTest, NonNullHandleUnloadsOnceAfterMoves) {
  ModuleHandle foo(reinterpret_cast<void*>(1));
  MockStreamExecutor executor;
  EXPECT_CALL(executor, UnloadModule).WillOnce(Return(true));
  {
    ScopedModuleHandle first(&executor, foo);
    ScopedModuleHandle second = std::move(first);
    ScopedModuleHandle third(&executor, ModuleHandle{});
    third = std::move(second);
  }
}

}  // namespace
}  // namespace stream_executor
