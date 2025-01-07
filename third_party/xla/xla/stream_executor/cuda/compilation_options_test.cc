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

#include "xla/stream_executor/cuda/compilation_options.h"

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/test.h"

namespace stream_executor::cuda {
namespace {

TEST(CompilationOptionsTest, HashAndComparison) {
  CompilationOptions options1;
  CompilationOptions options2;
  EXPECT_EQ(options1, options2);

  options2.disable_optimizations = true;
  options2.cancel_if_reg_spill = true;
  options2.generate_line_info = true;
  options2.generate_debug_info = true;
  EXPECT_NE(options1, options2);

  absl::flat_hash_set<CompilationOptions> options_set;
  options_set.insert(options1);
  options_set.insert(options2);
  options_set.insert(CompilationOptions{options2});
  EXPECT_EQ(options_set.size(), 2);
}

TEST(CompilationOptionsTest, Stringify) {
  CompilationOptions options;
  options.disable_optimizations = true;
  options.cancel_if_reg_spill = false;
  options.generate_line_info = true;
  options.generate_debug_info = false;
  EXPECT_EQ(absl::StrCat(options),
            "disable_optimizations: true, cancel_if_reg_spill: false, "
            "generate_line_info: true, generate_debug_info: false");
}

}  // namespace
}  // namespace stream_executor::cuda
