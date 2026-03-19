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

#include "xla/stream_executor/cuda/cuda_diagnostics.h"

#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

void EnsureCudaIsInitialized() {
  // Platform is intentionally leaked.
  // See the comment in platform_manager.h.
  absl::LeakCheckDisabler disabler;

  TF_ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                          PlatformManager::PlatformWithName("CUDA"));
  CHECK_GT(platform->VisibleDeviceCount(), 0);
}

TEST(CudaDiagnosticsTest, DiagnosticRuns) {
  // Initialize the platform - this is not code under test, it only ensures that
  // we have a working CUDA setup.
  EnsureCudaIsInitialized();

  cuda::Diagnostician::LogDiagnosticInformation();
}

}  // namespace
}  // namespace stream_executor::gpu
