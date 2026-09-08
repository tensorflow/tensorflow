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

#include "xla/stream_executor/rocm/delay_kernel.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

// The kernel gives up on the host after 100ms; see LaunchDelayKernel.
constexpr absl::Duration kKernelTimeout = absl::Milliseconds(100);

class DelayKernelTest : public ::testing::Test {
 public:
  // Launches the delay kernel and immediately releases it again, so that the
  // launch under test does not pay for loading the code object. That load
  // synchronises, which can stall the host for longer than the kernel is
  // willing to wait.
  void WarmUp() {
    TF_ASSERT_OK_AND_ASSIGN(GpuSemaphore semaphore,
                            LaunchDelayKernel(stream_.get()));
    *semaphore = GpuSemaphoreState::kRelease;
    ASSERT_THAT(stream_->BlockHostUntilDone(), absl_testing::IsOk());
  }

  StreamExecutor* executor_;
  std::unique_ptr<Stream> stream_;

 private:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        PlatformManager::PlatformWithId(rocm::kROCmPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
  }
};

// The kernel must hold the stream until the host releases it, and it must stop
// because of that release rather than because it timed out waiting for it.
// Only then do the timer's start event and the timed work land back to back,
// and it only happens if the device observes host writes to the semaphore at
// all, which depends on the memory it lives in. See CreateCoherentSemaphore.
TEST_F(DelayKernelTest, HostReleaseStopsTheKernelBeforeItTimesOut) {
  WarmUp();

  TF_ASSERT_OK_AND_ASSIGN(GpuSemaphore semaphore,
                          LaunchDelayKernel(stream_.get()));
  TF_ASSERT_OK_AND_ASSIGN(RocmEvent event,
                          RocmEvent::Create(executor_, /*allow_timing=*/false));
  ASSERT_THAT(stream_->RecordEvent(&event), absl_testing::IsOk());

  // The kernel is still spinning, so nothing recorded behind it can have
  // completed, and it has not given up yet either.
  EXPECT_EQ(event.PollForStatus(), Event::Status::kPending);
  ASSERT_EQ(*semaphore, GpuSemaphoreState::kHold);

  *semaphore = GpuSemaphoreState::kRelease;
  ASSERT_THAT(stream_->BlockHostUntilDone(), absl_testing::IsOk());

  // A kernel that stopped for any other reason than the release above would
  // have overwritten the semaphore with kTimedOut.
  EXPECT_EQ(*semaphore, GpuSemaphoreState::kRelease);
  EXPECT_EQ(event.PollForStatus(), Event::Status::kComplete);
}

// The other half of the contract. A host that never releases the kernel still
// gets the stream back, and can tell that the measurement was compromised.
TEST_F(DelayKernelTest, TimesOutWhenTheHostNeverReleasesIt) {
  WarmUp();

  TF_ASSERT_OK_AND_ASSIGN(GpuSemaphore semaphore,
                          LaunchDelayKernel(stream_.get()));
  absl::SleepFor(2 * kKernelTimeout);

  ASSERT_THAT(stream_->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_EQ(*semaphore, GpuSemaphoreState::kTimedOut);
}

}  // namespace
}  // namespace stream_executor::gpu
