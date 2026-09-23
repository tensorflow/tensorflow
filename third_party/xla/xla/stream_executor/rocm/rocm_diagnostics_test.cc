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

#include "xla/stream_executor/rocm/rocm_diagnostics.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"

namespace stream_executor::gpu {
namespace {

void EnsureRocmIsInitialized() {
  // Platform is intentionally leaked.
  // See the comment in platform_manager.h.
  absl::LeakCheckDisabler disabler;

  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);
}

TEST(RocmDiagnosticsTest, DiagnosticRuns) {
  // Platform init is not under test; it only provides a working ROCm context.
  ASSERT_NO_FATAL_FAILURE(EnsureRocmIsInitialized());

  rocm::LogDiagnosticInformation();
}

}  // namespace
}  // namespace stream_executor::gpu
