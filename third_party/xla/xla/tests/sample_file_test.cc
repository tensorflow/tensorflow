/* Copyright 2017 The OpenXLA Authors.

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

// This demonstrates how to use hlo_test_base to create a file based testcase
// and compare results on gpu and cpu.

#include <string>
#include <vector>

#include "xla/hlo/testlib/test.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class SampleFileTest : public HloTestBase {
 protected:
  SampleFileTest()
      : HloTestBase(
            /*test_platform=*/PlatformUtil::GetPlatform("gpu").value(),
            /*reference_platform=*/PlatformUtil::GetPlatform("cpu").value()) {}
};

TEST_F(SampleFileTest, Convolution) {
  const std::string& filename = tsl::io::JoinPath(
      tsl::testing::XlaSrcRoot(), "tests", "isolated_convolution.hlo");
  EXPECT_TRUE(RunAndCompareFromFile(filename, ErrorSpec{0.01}));
}

}  // namespace
}  // namespace xla
