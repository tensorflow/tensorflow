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

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/libraries/cub/generate_cub_scratch_size_lib.cu.h"

namespace xla::gpu {
namespace {

TEST(ExtractCubScratchSizeTest, ExtractData) {
  ASSERT_OK(GenerateCubScratchSizeData());
}

}  // namespace
}  // namespace xla::gpu
