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

#include "xla/service/gpu/gpu_asm_opts_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/xla.pb.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {
using ::testing::Contains;

TEST(PtxOptsFromDebugOptionsTest, GenerateLineInfo) {
  xla::DebugOptions debug_options;
  debug_options.set_xla_gpu_generate_line_info(true);

  EXPECT_THAT(PtxOptsFromDebugOptions(debug_options).extra_flags,
              Contains("--generate-line-info"));
}

TEST(PtxOptsFromDebugOptionsTest, GenerateDebugInfo) {
  xla::DebugOptions debug_options;
  debug_options.set_xla_gpu_generate_debug_info(true);

  EXPECT_THAT(PtxOptsFromDebugOptions(debug_options).extra_flags,
              Contains("--device-debug"));
}

}  // namespace
}  // namespace xla::gpu
