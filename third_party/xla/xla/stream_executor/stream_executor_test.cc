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

#include "xla/stream_executor/stream_executor.h"

#include <memory>

#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {

static std::unique_ptr<StreamExecutor> NewStreamExecutor() {
  Platform* platform = PlatformManager::PlatformWithName("Host").value();
  StreamExecutorConfig config(/*ordinal=*/0);
  return platform->GetUncachedExecutor(config).value();
}

TEST(StreamExecutorTest, HostMemoryAllocate) {
  auto executor = NewStreamExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto allocation, executor->HostMemoryAllocate(1024));
  EXPECT_NE(allocation->opaque(), nullptr);
  EXPECT_EQ(allocation->size(), 1024);
}

}  // namespace stream_executor
