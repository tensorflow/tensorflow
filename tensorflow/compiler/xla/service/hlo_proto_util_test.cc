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
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace {

class HloProtoUtilTest : public ::testing::Test {};

TEST_F(HloProtoUtilTest, ParamsAndOutputShape) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_entry_computation_name("entry");
  HloComputationProto* computation = module->add_computations();
  computation->set_name("entry");
  computation->set_root_name("root");

  HloInstructionProto* param0 = computation->add_instructions();
  param0->set_opcode(HloOpcodeString(HloOpcode::kParameter));
  param0->set_parameter_number(0);
  *param0->mutable_shape() = ShapeUtil::MakeShape(F32, {42});

  HloInstructionProto* param2 = computation->add_instructions();
  param2->set_opcode(HloOpcodeString(HloOpcode::kParameter));
  param2->set_parameter_number(2);
  *param2->mutable_shape() = ShapeUtil::MakeShape(S32, {1, 2, 3});

  HloInstructionProto* param1 = computation->add_instructions();
  param1->set_opcode(HloOpcodeString(HloOpcode::kParameter));
  param1->set_parameter_number(1);
  *param1->mutable_shape() = ShapeUtil::MakeShape(F64, {});

  HloInstructionProto* root = computation->add_instructions();
  root->set_opcode(HloOpcodeString(HloOpcode::kAdd));
  root->set_name("root");
  *root->mutable_shape() = ShapeUtil::MakeShape(U8, {2});

  VLOG(1) << hlo_proto.DebugString();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<const Shape*> parameter_shapes,
                          EntryComputationParameterShapes(hlo_proto));
  ASSERT_EQ(parameter_shapes.size(), 3);
  EXPECT_TRUE(
      ShapeUtil::Equal(*parameter_shapes[0], ShapeUtil::MakeShape(F32, {42})));
  EXPECT_TRUE(
      ShapeUtil::Equal(*parameter_shapes[1], ShapeUtil::MakeShape(F64, {})));
  EXPECT_TRUE(ShapeUtil::Equal(*parameter_shapes[2],
                               ShapeUtil::MakeShape(S32, {1, 2, 3})));

  TF_ASSERT_OK_AND_ASSIGN(const Shape* output_shape,
                          EntryComputationOutputShape(hlo_proto));
  EXPECT_TRUE(ShapeUtil::Equal(*output_shape, ShapeUtil::MakeShape(U8, {2})));
}

TEST_F(HloProtoUtilTest, ParamsAndOutputShapeNoParameters) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_entry_computation_name("entry");
  HloComputationProto* computation = module->add_computations();
  computation->set_name("entry");
  computation->set_root_name("root");

  HloInstructionProto* root = computation->add_instructions();
  root->set_opcode(HloOpcodeString(HloOpcode::kAdd));
  root->set_name("root");
  *root->mutable_shape() = ShapeUtil::MakeShape(U8, {2});

  TF_ASSERT_OK_AND_ASSIGN(std::vector<const Shape*> parameter_shapes,
                          EntryComputationParameterShapes(hlo_proto));
  ASSERT_EQ(parameter_shapes.size(), 0);
}

TEST_F(HloProtoUtilTest, ParamsAndOutputShapeMissingModule) {
  HloProto hlo_proto;

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("missing HloModuleProto"));
}

TEST_F(HloProtoUtilTest, ParamsAndOutputShapeMissingEntryComputation) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_entry_computation_name("entry");
  HloComputationProto* computation = module->add_computations();
  computation->set_name("not_entry");

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("has no entry computation named"));
}

TEST_F(HloProtoUtilTest, OutputShapeMissingEntryRoot) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_entry_computation_name("entry");
  HloComputationProto* computation = module->add_computations();
  computation->set_name("entry");
  computation->set_root_name("root");

  auto status = EntryComputationOutputShape(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("has no instruction named"));
}

TEST_F(HloProtoUtilTest, ParamsShapesMissingParameterNumbers) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_entry_computation_name("entry");
  HloComputationProto* computation = module->add_computations();
  computation->set_name("entry");
  computation->set_root_name("root");

  HloInstructionProto* param0 = computation->add_instructions();
  param0->set_opcode(HloOpcodeString(HloOpcode::kParameter));
  param0->set_parameter_number(0);
  *param0->mutable_shape() = ShapeUtil::MakeShape(F32, {42});

  HloInstructionProto* param2 = computation->add_instructions();
  param2->set_opcode(HloOpcodeString(HloOpcode::kParameter));
  param2->set_parameter_number(2);
  *param2->mutable_shape() = ShapeUtil::MakeShape(S32, {1, 2, 3});

  HloInstructionProto* root = computation->add_instructions();
  root->set_opcode(HloOpcodeString(HloOpcode::kAdd));
  root->set_name("root");
  *root->mutable_shape() = ShapeUtil::MakeShape(U8, {2});

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.error_message(),
              ::testing::HasSubstr("invalid parameter number"));
}

}  // namespace
}  // namespace xla
