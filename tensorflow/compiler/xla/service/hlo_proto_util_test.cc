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

#include "tensorflow/compiler/xla/service/hlo_proto_util.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

class HloProtoUtilTest : public ::testing::Test {};

TEST_F(HloProtoUtilTest, ParamsAndOutputShapeMissingModule) {
  HloProto hlo_proto;

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("missing HloModuleProto"));
}

TEST_F(HloProtoUtilTest, MissingProgramShape) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_name("entry");

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("missing program shape"));
}

}  // namespace
}  // namespace xla
