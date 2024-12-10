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

#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {

absl::StatusOr<std::vector<uint8_t>> GetGpuTestKernelsFatbin() {
  tsl::Env* env = tsl::Env::Default();
  std::string file_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "stream_executor", "gpu",
                        "gpu_test_kernels.fatbin");
  std::string file_contents;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, file_path, &file_contents));
  return std::vector<uint8_t>(file_contents.begin(), file_contents.end());
}
}  // namespace stream_executor::gpu
