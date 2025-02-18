/* Copyright 2022 The TensorFlow Authors All Rights Reserved.

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

#include "absl/status/statusor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/profiler/lib/profiler_lock.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(ProfilerDisabledTest, ProfilerDisabledTest) {
  setenv("TF_DISABLE_PROFILING", "1", /*overwrite=*/1);
  absl::StatusOr<tsl::profiler::ProfilerLock> profiler_lock =
      tsl::profiler::ProfilerLock::Acquire();
  EXPECT_FALSE(profiler_lock.ok());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
