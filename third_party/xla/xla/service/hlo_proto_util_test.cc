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

#include "xla/service/hlo_proto_util.h"

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"

namespace xla {
namespace {

class HloProtoUtilTest : public ::testing::Test {};

TEST_F(HloProtoUtilTest, ParamsAndOutputShapeMissingModule) {
  HloProto hlo_proto;

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), ::testing::HasSubstr("missing HloModuleProto"));
}

TEST_F(HloProtoUtilTest, MissingProgramShape) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_name("entry");

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), ::testing::HasSubstr("missing program shape"));
}

TEST_F(HloProtoUtilTest, GetBackendConfigStringPayloadValue) {
  HloInstructionProto instruction;
  instruction.mutable_backend_config_payload()->set_value("inline_config");
  auto statusor = GetBackendConfigString(instruction);
  ASSERT_TRUE(statusor.ok());
  EXPECT_EQ(statusor.value(), "inline_config");
}

TEST_F(HloProtoUtilTest, GetBackendConfigStringPayloadId) {
  HloInstructionProto instruction;
  instruction.mutable_backend_config_payload()->set_id(1);

  HloModuleProto module;
  module.add_payloads("payload_0");
  module.add_payloads("payload_1");

  auto statusor = GetBackendConfigString(instruction, &module);
  ASSERT_TRUE(statusor.ok());
  EXPECT_EQ(statusor.value(), "payload_1");
}

TEST_F(HloProtoUtilTest, GetBackendConfigStringMissingPayloadFallback) {
  HloInstructionProto instruction;
  instruction.set_backend_config("legacy_config");

  auto statusor = GetBackendConfigString(instruction);
  ASSERT_TRUE(statusor.ok());
  EXPECT_EQ(statusor.value(), "legacy_config");
}

TEST_F(HloProtoUtilTest, GetBackendConfigStringInvalidId) {
  HloInstructionProto instruction;
  instruction.mutable_backend_config_payload()->set_id(5);

  HloModuleProto module;
  module.add_payloads("payload_0");

  auto statusor = GetBackendConfigString(instruction, &module);
  ASSERT_FALSE(statusor.ok());
  ASSERT_THAT(statusor.status().message(),
              ::testing::HasSubstr("Payload requested ID"));
}

}  // namespace
}  // namespace xla
