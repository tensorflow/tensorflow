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

#include "xla/service/gpu/fusion_process_dump.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/statusor.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {
namespace {

using FusionProcessDumpTest = HloHardwareIndependentTestBase;

void AddFusion(FusionProcessDumpProto& dump_proto,
               const std::string& fusion_name, const std::string& producer_name,
               const std::string& consumer_name) {
  auto step = dump_proto.add_fusion_steps();
  auto fusion_step = step->mutable_fusion();
  fusion_step->set_fusion_name(fusion_name);
  fusion_step->set_producer_name(producer_name);
  fusion_step->set_consumer_name(consumer_name);
}

TEST_F(FusionProcessDumpTest, MultipleFusionSteps) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      add = f32[] add(p0, p1)
      subtract = f32[] subtract(p0, p1)
      abs = f32[] abs(subtract)
      ROOT multiply = f32[] multiply(add, abs)
  })"));

  FusionProcessDumpProto dump_proto;
  *dump_proto.mutable_gpu_device_info() =
      TestGpuDeviceInfo::RTXA6000DeviceInfo().ToGpuProto();
  dump_proto.set_hlo_module_before_fusion(
      module->ToString(HloPrintOptions::ShortParsable()));

  AddFusion(dump_proto, "fusion.1", "subtract", "abs");
  AddFusion(dump_proto, "fusion.2", "fusion.1", "multiply");
  AddFusion(dump_proto, "fusion.2", "add", "fusion.2");

  TF_ASSERT_OK_AND_ASSIGN(auto fusion_process_dump,
                          FusionProcessDump::LoadFromProto(dump_proto));

  fusion_process_dump.Advance();
  fusion_process_dump.Advance();
  fusion_process_dump.Advance();

  EXPECT_FALSE(fusion_process_dump.HasNext());

  auto root =
      fusion_process_dump.module()->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "fusion.2");
  ASSERT_THAT(root, GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Multiply(
                  m::Add(m::Parameter(), m::Parameter()),
                  m::Abs(m::Subtract(m::Parameter(), m::Parameter())))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
