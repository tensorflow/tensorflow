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

  TF_ASSERT_OK_AND_ASSIGN(HloModuleGroup group_copy,
                          HloModuleGroup::CreateFromProto(
                              group.ToProto(), {group.module(0).config()}));
  EXPECT_EQ(group_copy.modules().size(), 1);
  EXPECT_THAT(
      group_copy.module(0).entry_computation()->instructions(),
      ::testing::ElementsAre(op::Parameter(), op::Parameter(), op::Add()));

  std::vector<std::unique_ptr<HloModule>> modules = group.ConsumeModules();
  EXPECT_EQ(modules.size(), 1);
}

// Test that metadata is transferred when a module is replaced.
TEST_F(HloModuleGroupTest, ReplaceModuleMetadata) {
  auto old_module = CreateNewVerifiedModule();
  int old_module_id = old_module->unique_id();
  old_module->metadata()->RecordPassStart();
  TF_EXPECT_OK(old_module->metadata()->set_current_pass_name("fake pass"));

  HloModuleGroup group(std::move(old_module));
  EXPECT_EQ(group.module(0).metadata()->proto().module_group_name(),
            group.name());

  auto new_module = CreateNewVerifiedModule();
  group.ReplaceModule(0, std::move(new_module));

  EXPECT_NE(group.module(0).unique_id(), old_module_id);
  const HloModuleMetadataProto& module_metadata =
      group.module(0).metadata()->proto();
  EXPECT_EQ(module_metadata.canonical_module_id(), old_module_id);

  const HloPassMetadata& pass_metadata =
      *module_metadata.pass_metadata().rbegin();
  EXPECT_THAT(pass_metadata,
              Property(&HloPassMetadata::pass_name, StrEq("fake pass")));
}

}  // namespace

}  // namespace xla
