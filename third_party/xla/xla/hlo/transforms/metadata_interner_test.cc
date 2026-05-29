/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/metadata_interner.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

using MetadataInternerTest = HloHardwareIndependentTestBase;

TEST_F(MetadataInternerTest, ExtractionAndDeduplication) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder("comp");
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));

  p0->set_frontend_attribute("xla_interned_metadata", "tokamax:{\"data\": 1}");
  p1->set_frontend_attribute("xla_interned_metadata", "tokamax:{\"data\": 1}");

  module->AddEntryComputation(builder.Build());

  MetadataInterner interner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, interner.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_FALSE(
      p0->frontend_attributes().map().contains("xla_interned_metadata"));
  EXPECT_FALSE(
      p1->frontend_attributes().map().contains("xla_interned_metadata"));
  EXPECT_TRUE(p0->has_interned_metadata());
  EXPECT_TRUE(p1->has_interned_metadata());
  EXPECT_TRUE(p0->metadata().interned_metadata_payload().has_value());
  EXPECT_TRUE(p1->metadata().interned_metadata_payload().has_value());
  EXPECT_FALSE(p0->metadata().interned_metadata_payload().has_id());
  EXPECT_FALSE(p1->metadata().interned_metadata_payload().has_id());

  EXPECT_EQ(p0->interned_metadata_string(), "tokamax:{\"data\": 1}");
  EXPECT_EQ(p1->interned_metadata_string(), "tokamax:{\"data\": 1}");

  // Verify that serialization-time deduplication works flawlessly!
  HloModuleProto proto = module->ToProto(HloProtoOptions(false, true));
  EXPECT_EQ(proto.payloads_size(), 1);
  EXPECT_EQ(proto.payloads(0), "tokamax:{\"data\": 1}");

  const HloInstructionProto& p0_proto = proto.computations(0).instructions(0);
  const HloInstructionProto& p1_proto = proto.computations(0).instructions(1);
  EXPECT_EQ(p0_proto.metadata().interned_metadata_payload().id(), 0);
  EXPECT_EQ(p1_proto.metadata().interned_metadata_payload().id(), 0);
}

}  // namespace
}  // namespace xla
