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

#include "xla/stream_executor/cuda/nvjitlink_known_issues.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "tsl/platform/test.h"

namespace {

TEST(NvJitLinkKnownIssuesTest, ReturnsFalseWhenNvJitLinkIsNotAvailable) {
  if (stream_executor::IsLibNvJitLinkSupported()) {
    GTEST_SKIP();
  }
  // This is the only invariance we can test without writing a pointless change
  // detector test.
  EXPECT_FALSE(stream_executor::LoadedNvJitLinkHasKnownIssues());
}

}  // namespace
