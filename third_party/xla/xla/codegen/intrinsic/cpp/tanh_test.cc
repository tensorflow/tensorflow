/* Copyright 2025 The OpenXLA Authors.

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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/cpp/tanh_ll.h"

namespace xla {
namespace codegen {
using ::testing::ContainsRegex;

namespace {

TEST(TanhTest, FloatTanhVectorized) {
  std::string ir = llvm_ir::kTanhLlIr;
  EXPECT_THAT(ir, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(
      ir, ContainsRegex("fcmp olt <4 x float>.*float 0x3F3A36E2E0000000.*"));
}
}  // namespace
}  // namespace codegen
}  // namespace xla
