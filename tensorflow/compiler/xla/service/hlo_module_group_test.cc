/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module_group.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module_group_metadata.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {

namespace {

namespace op = ::xla::testing::opcode_matchers;

class HloModuleGroupTest : public HloTestBase {
 protected:
  HloModuleGroupTest() = default;
};

TEST_F(HloModuleGroupTest, SingleModule) {
  const string text = R"(
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
  EXPECT_EQ(group.modules().size(), 0);
}

TEST_F(HloModuleGroupTest, MultipleModules) {
  const string text_0 = R"(
HloModule module0

ENTRY %entry (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}
)";
  const string text_1 = R"(
HloModule module1

ENTRY %entry (a: f32[]) -> f32[] {
  ROOT %a = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_0,
                          ParseAndReturnVerifiedModule(text_0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_1,
                          ParseAndReturnVerifiedModule(text_1));
  std::vector<std::unique_ptr<HloModule>> modules;
  modules.push_back(std::move(module_0));
  modules.push_back(std::move(module_1));
  HloModuleGroup group(TestName(), absl::MakeSpan(modules));
  EXPECT_EQ(group.modules().size(), 2);
  EXPECT_THAT(
      group.module(0).entry_computation()->instructions(),
      ::testing::ElementsAre(op::Parameter(), op::Parameter(), op::Add()));
  EXPECT_THAT(group.module(1).entry_computation()->instructions(),
              ::testing::ElementsAre(op::Parameter()));

  TF_ASSERT_OK_AND_ASSIGN(HloModuleGroup group_copy,
                          HloModuleGroup::CreateFromProto(
                              group.ToProto(), {group.module(0).config(),
                                                group.module(1).config()}));
  EXPECT_EQ(group_copy.modules().size(), 2);
}

TEST_F(HloModuleGroupTest, BuildModuleGroupByPushBack) {
  const string text_0 = R"(
HloModule module0

ENTRY %entry (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}
)";
  const string text_1 = R"(
HloModule module1

ENTRY %entry (a: f32[]) -> f32[] {
  ROOT %a = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_0,
                          ParseAndReturnVerifiedModule(text_0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_1,
                          ParseAndReturnVerifiedModule(text_1));
  HloModuleGroup group(TestName());
  group.push_back(std::move(module_0));
  group.push_back(std::move(module_1));

  EXPECT_EQ(group.modules().size(), 2);
  EXPECT_THAT(
      group.module(0).entry_computation()->instructions(),
      ::testing::ElementsAre(op::Parameter(), op::Parameter(), op::Add()));
  EXPECT_THAT(group.module(1).entry_computation()->instructions(),
              ::testing::ElementsAre(op::Parameter()));
}

// Tests that the order of companion instructions in the companion set doesn't
// change across runs.
TEST_F(HloModuleGroupTest, ModuleGroupCompanionOrder) {
  // A simple while loop template for core i sending to core i+1.
  constexpr char text[] = R"(
HloModule module_%d

while_cond {
  param = s32[] parameter(0)
  ROOT p = pred[] constant(true)
}

while_body {
  param = s32[] parameter(0)
  token.s = token[] after-all()
  token.r = token[] after-all()
  send = (s32[], u32[], token[]) send(param, token.s), channel_id=%d
  send-done = token[] send-done(send), channel_id=%d
  recv = (s32[], u32[], token[]) recv(token.r), channel_id=%d
  recv-done = (s32[], token[]) recv-done(recv), channel_id=%d
  ROOT data = s32[] get-tuple-element(recv-done), index=0
}

ENTRY entry {
  while_init = s32[] constant(1)
  ROOT while = s32[] while(while_init), condition=while_cond, body=while_body
}
)";

  // Try creating the module and the metadata kTrialCount times and check the
  // companion instructions remain in the same order.
  const int64 kTrialCount = 5;
  const int64 kDeviceCount = 10;
  std::vector<int64> companion_order;

  for (int64 t = 0; t < kTrialCount; ++t) {
    HloModuleGroup group(TestName());
    for (int64 i = 0; i < kDeviceCount; ++i) {
      const int64 send_channel = i;
      const int64 recv_channel = i == 0 ? kDeviceCount - 1 : i - 1;
      TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                              ParseAndReturnVerifiedModule(absl::StrFormat(
                                  text, i, send_channel, send_channel,
                                  recv_channel, recv_channel)));
      group.push_back(std::move(module));
    }
    ASSERT_EQ(group.modules().size(), kDeviceCount);

    TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                            HloModuleGroupMetadata::Build(group.modules()));
    ASSERT_EQ(metadata->companion_sets().size(), 1);

    std::vector<int64> module_ids;
    for (HloInstruction* companion : *metadata->companion_sets()[0]) {
      module_ids.push_back(metadata->GetModuleId(companion->GetModule()));
    }

    if (t == 0) {
      companion_order = module_ids;
    } else {
      EXPECT_TRUE(absl::c_equal(companion_order, module_ids));
    }
  }
}

}  // namespace

}  // namespace xla
