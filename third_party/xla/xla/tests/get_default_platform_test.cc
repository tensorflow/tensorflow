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

#include "absl/status/status.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

// Although we don't use any of the functionality provided by HloPjRtTestBase,
// we want to model the same environment as a PjRt migrated test that ends up
// calling GetDefaultPlatform.
using GetDefaultPlatformTest = HloPjRtTestBase;

// Regression test to ensure that it's not possible to call GetDefaultPlatform
// on a PJRT migrated test due to
// --xla_allow_get_default_platform=false being set in XLA_FLAGS.
TEST_F(GetDefaultPlatformTest, GetDefaultPlatformFails) {
  EXPECT_THAT(PlatformUtil::GetDefaultPlatform(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("allow_get_default_platform")));
}

}  // namespace
}  // namespace xla
