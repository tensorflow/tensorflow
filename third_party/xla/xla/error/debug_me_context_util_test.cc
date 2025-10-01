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

#include "xla/error/debug_me_context_util.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/debug_me_context.h"

namespace xla {
namespace {

TEST(DebugMeContextUtil, StringCheck) {
  constexpr absl::string_view kCompilerName{"MyCompiler"};

  tsl::DebugMeContext<error::DebugMeContextKey> ctx(
      error::DebugMeContextKey::kCompiler, std::string(kCompilerName));

  const std::string error_message =
      error::DebugMeContextToErrorMessageString();

  EXPECT_TRUE(absl::StrContains(error_message, kCompilerName));
}

}  // namespace
}  // namespace xla
