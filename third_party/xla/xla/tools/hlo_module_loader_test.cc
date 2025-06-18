/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/tools/hlo_module_loader.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class HloModuleLoaderTest : public HloHardwareIndependentTestBase {};

TEST_F(HloModuleLoaderTest, StripsLogHeaders) {
  const std::string& hlo_string = R"(
I0521 12:04:45.883483    1509 service.cc:186] HloModule test_log_stripping
I0521 12:04:45.883483    1509 service.cc:186]
I0521 12:04:45.883483    1509 service.cc:186] ENTRY entry {
I0521 12:04:45.883483    1509 service.cc:186]   p0 = f32[4]{0} parameter(0)
I0521 12:04:45.883483    1509 service.cc:186]   p1 = f32[4]{0} parameter(1)
I0521 12:04:45.883483    1509 service.cc:186]   add = f32[4]{0} add(p0, p1)
I0521 12:04:45.883483    1509 service.cc:186]   ROOT rooty = (f32[4]{0}, f32[4]{0}) tuple(p1, add)
I0521 12:04:45.883483    1509 service.cc:186] }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          LoadModuleFromData(hlo_string, "txt"));
  EXPECT_NE(FindInstruction(hlo_module.get(), "p0"), nullptr);
  EXPECT_NE(FindInstruction(hlo_module.get(), "p1"), nullptr);
  EXPECT_NE(FindInstruction(hlo_module.get(), "add"), nullptr);
  EXPECT_NE(FindInstruction(hlo_module.get(), "rooty"), nullptr);
}

}  // namespace
}  // namespace xla
