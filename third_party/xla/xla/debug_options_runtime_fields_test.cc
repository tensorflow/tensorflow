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

#include "xla/debug_options_runtime_fields.h"

#include <gtest/gtest.h>
#include "xla/xla.pb.h"

namespace xla {
namespace {

TEST(DebugOptionsRuntimeFieldsTest, ListsFieldsThatChangeExecution) {
  // Read by the NaN checker thunk pass when the executable is loaded.
  EXPECT_TRUE(
      IsRuntimeDebugOptionsField(DebugOptions::kXlaGpuDetectNanFieldNumber));
  // Read from the process's flags when the client is created.
  EXPECT_TRUE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaForceHostPlatformDeviceCountFieldNumber));
  // Read by the watchdog while the executable runs.
  EXPECT_TRUE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaGpuDeviceExecutionTerminateTimeoutFieldNumber));
  // An empty value makes loading fail with the NaN checker in DUMP mode.
  EXPECT_TRUE(IsRuntimeDebugOptionsField(DebugOptions::kXlaDumpToFieldNumber));
}

TEST(DebugOptionsRuntimeFieldsTest, OmitsCompileOnlyAndDiagnosticsFields) {
  // Read only while compiling.
  EXPECT_FALSE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaDebugBufferAssignmentShowMaxFieldNumber));
  EXPECT_FALSE(IsRuntimeDebugOptionsField(
      DebugOptions::
          kXlaGpuExperimentalEnableDynamicSliceExtendedOffsetsFieldNumber));
  // Read after compilation, but only changes which dump files are written.
  EXPECT_FALSE(
      IsRuntimeDebugOptionsField(DebugOptions::kXlaDumpHloAsTextFieldNumber));
  // Read after compilation, but only affects profiling.
  EXPECT_FALSE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaGpuTraceAnnotationLevelFieldNumber));
  EXPECT_FALSE(IsRuntimeDebugOptionsField(
      DebugOptions::kXlaGpuEnableCuptiMultiSubscriberFieldNumber));
}

}  // namespace
}  // namespace xla
