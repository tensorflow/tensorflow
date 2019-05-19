/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/ops.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(OpsTest, TestBasicOpRegistration) {
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SomeOp");
  TF_OpDefinitionBuilderAddStringAttr(builder, "attr1");
  TF_OpDefinitionBuilderAddInput(builder, "input1", TF_UINT8);
  TF_OpDefinitionBuilderAddInput(builder, "input2", TF_UINT16);
  TF_OpDefinitionBuilderAddOutput(builder, "output1", TF_UINT32);
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_Buffer* op_list_buffer = TF_GetAllOpList();
  ::tensorflow::OpList op_list;
  op_list.ParseFromArray(op_list_buffer->data, op_list_buffer->length);
  bool found = false;
  for (const auto& op : op_list.op()) {
    if (op.name() == "SomeOp") {
      ASSERT_EQ(2, op.input_arg_size());
      ASSERT_EQ("input1", op.input_arg(0).name());
      ASSERT_EQ(::tensorflow::DT_UINT8, op.input_arg(0).type());
      ASSERT_EQ(1, op.attr_size());
      ASSERT_EQ("string", op.attr(0).type());
      found = true;
    }
  }
  EXPECT_TRUE(found);
  TF_DeleteStatus(status);
  TF_DeleteBuffer(op_list_buffer);
}

void identity_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
}

TEST(OpsTest, TestShapeInference_IdentityFunction) {
  ShapeInferenceTestOp op("SomeTestOp");

  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SomeTestOp");
  TF_OpDefinitionBuilderAddInput(builder, "input1", TF_UINT8);
  TF_OpDefinitionBuilderAddOutput(builder, "output1", TF_UINT8);
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &identity_shape_fn);
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_ASSERT_OK(
      shape_inference::ShapeInferenceTestutil::InferShapes(op, "[1,2]", "in0"));
  TF_DeleteStatus(status);
}

// Creates an output whose shape is a vector of length
// TF_ShapeInferenceContextRank.
void vectorize_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));
  TF_ShapeHandle* new_shape = TF_ShapeInferenceContextVectorFromSize(
      ctx, TF_ShapeInferenceContextRank(ctx, handle));
  TF_ShapeInferenceContextSetOutput(ctx, 0, new_shape, status);
  TF_DeleteShapeHandle(handle);
  TF_DeleteShapeHandle(new_shape);
}

TEST(OpsTest, TestShapeInference_VectorizeFunction) {
  ShapeInferenceTestOp op("VectorizeTestOp");

  TF_OpDefinitionBuilder* builder =
      TF_NewOpDefinitionBuilder("VectorizeTestOp");
  TF_OpDefinitionBuilderAddInput(builder, "input1", TF_UINT8);
  TF_OpDefinitionBuilderAddOutput(builder, "output1", TF_UINT8);
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &vectorize_shape_fn);
  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_ASSERT_OK(shape_inference::ShapeInferenceTestutil::InferShapes(
      op, "[4,5,9]", "[3]"));
  TF_DeleteStatus(status);
}

TEST(OpsTest, AttributeAccessors) {
  TF_OpDefinitionBuilder* builder =
      TF_NewOpDefinitionBuilder("AttributeAccesorsOp");
  float values[] = {1, 2, 3, 4};
  TF_OpDefinitionBuilderAddFloatListAttrWithDefaultValues(
      builder, "foo1", values, sizeof(values));
  TF_OpDefinitionBuilderAddStringAttrWithDefaultValue(builder, "foo2",
                                                      "my string");
  TF_OpDefinitionBuilderSetIsCommutative(builder, true);
  TF_OpDefinitionBuilderSetIsAggregate(builder, true);
  TF_OpDefinitionBuilderSetAllowsUninitializedInput(builder, true);
  std::string deprecation_msg = "use something else instead";
  TF_OpDefinitionBuilderDeprecated(builder, 4, deprecation_msg.c_str());

  TF_Status* status = TF_NewStatus();
  TF_RegisterOpDefinition(builder, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status));

  TF_Buffer* op_list_buffer = TF_GetAllOpList();
  ::tensorflow::OpList op_list;
  op_list.ParseFromArray(op_list_buffer->data, op_list_buffer->length);
  bool found = false;
  for (const auto& op : op_list.op()) {
    if (op.name() == "AttributeAccesorsOp") {
      ASSERT_TRUE(op.is_commutative());
      ASSERT_TRUE(op.is_aggregate());
      ASSERT_TRUE(op.allows_uninitialized_input());
      ASSERT_EQ(4, op.deprecation().version());
      ASSERT_EQ(deprecation_msg, op.deprecation().explanation());
      ASSERT_EQ(2, op.attr_size());
      ASSERT_EQ("list(float)", op.attr(0).type());
      AttrValue::ListValue l = op.attr(0).default_value().list();
      ASSERT_EQ(1, l.f(0));
      ASSERT_EQ(2, l.f(1));
      ASSERT_EQ(3, l.f(2));
      ASSERT_EQ(4, l.f(3));

      ASSERT_EQ("string", op.attr(1).type());
      ASSERT_EQ("my string", op.attr(1).default_value().s());
      found = true;
    }
  }
  ASSERT_TRUE(found);
  TF_DeleteStatus(status);
  TF_DeleteBuffer(op_list_buffer);
}

}  // namespace
}  // namespace tensorflow
