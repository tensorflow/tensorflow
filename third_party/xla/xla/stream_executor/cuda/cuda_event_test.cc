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

#include "xla/stream_executor/cuda/cuda_event.h"

#include <utility>

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

TEST(CudaEventTest, CreateEvent) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::cuda::kCudaPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  CudaExecutor* cuda_executor = reinterpret_cast<CudaExecutor*>(executor);

  TF_ASSERT_OK_AND_ASSIGN(CudaEvent event,
                          CudaEvent::Create(cuda_executor, false));

  EXPECT_NE(event.GetHandle(), nullptr);
  EXPECT_EQ(event.PollForStatus(), Event::Status::kComplete);

  CUevent handle = event.GetHandle();
  CudaEvent event2 = std::move(event);
  EXPECT_EQ(event2.GetHandle(), handle);
}

}  // namespace

}  // namespace stream_executor::gpu
