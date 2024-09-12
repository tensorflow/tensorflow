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

#include "absl/status/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {

static absl::StatusOr<StreamExecutor*> NewStreamExecutor() {
  TF_ASSIGN_OR_RETURN(auto platform, PlatformManager::PlatformWithName("Host"));
  TF_ASSIGN_OR_RETURN(auto stream_exec, platform->ExecutorForDevice(0));
  return stream_exec;
}

TEST(StreamExecutorTest, HostMemoryAllocate) {
  TF_ASSERT_OK_AND_ASSIGN(auto executor, NewStreamExecutor());
  TF_ASSERT_OK_AND_ASSIGN(auto allocation, executor->HostMemoryAllocate(1024));
  EXPECT_NE(allocation->opaque(), nullptr);
  EXPECT_EQ(allocation->size(), 1024);
}

}  // namespace stream_executor
