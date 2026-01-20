/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_module_group.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {

namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::Property;
using ::testing::StrEq;

class HloModuleGroupTest : public HloHardwareIndependentTestBase {
 protected:
  HloModuleGroupTest() = default;
};

TEST_F(HloModuleGroupTest, SingleModule) {
  const std::string text = R"(
HloModule simple_module

ENTRY %entry (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(text));
  HloModuleGroup group(std::move(module));

  EXPECT_EQ(group.modules().size(), 1);
  EXPECT_THAT(
      group.module(0).entry_computation()->instructions(),
      ::testing::ElementsAre(op::Parameter(), op::Parameter(), op::Add()));


  std::vector<std::unique_ptr<HloModule>> modules = group.ConsumeModules();
  EXPECT_EQ(modules.size(), 1);
}
}  // namespace

}  // namespace xla
