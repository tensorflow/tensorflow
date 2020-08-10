/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(StateOpsTest, Assign_ShapeFn) {
  ShapeInferenceTestOp op("Assign");

  TF_ASSERT_OK(NodeDefBuilder("test", "Assign")
                   .Input("ref", 0, DT_FLOAT_REF)
                   .Input("value", 1, DT_FLOAT)
                   .Attr("validate_shape", true)
                   .Finalize(&op.node_def));
  INFER_OK(op, "[1,2];[1,2]", "in0");

  // Resolves shapes when validate_shape is True.
  INFER_OK(op, "[1,?];[?,2]", "[d0_0,d1_1]");

  // validate_shape=True, fails when the shapes are not compatible.
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 3", op,
              "[1,?];[3,2]");

  // Test for validate_shape=False
  TF_ASSERT_OK(NodeDefBuilder("test", "Assign")
                   .Input("ref", 0, DT_FLOAT_REF)
                   .Input("value", 1, DT_FLOAT)
                   .Attr("validate_shape", false)
                   .Finalize(&op.node_def));
  INFER_OK(op, "[1,2];[1,2,3,4]", "in1");
}

TEST(StateOpsTest, ScatterUpdate_ShapeFn) {
  ShapeInferenceTestOp op("ScatterUpdate");
  TF_ASSERT_OK(NodeDefBuilder("test", "ScatterUpdate")
                   .Input("ref", 0, DT_FLOAT_REF)
                   .Input("indices", 0, DT_INT32)
                   .Input("updates", 1, DT_FLOAT)
                   .Finalize(&op.node_def));
  INFER_OK(op, "[1,2];[3];[3,2]", "in0");

  // Resolve shape on first updates dimension.
  INFER_OK(op, "[1,2];[3];[?,2]", "in0");

  // Allow the update to be a scalar.
  INFER_OK(op, "[1,2];[3];?", "in0");

  // Allow a scalar index.
  INFER_OK(op, "[1,2];[];[2]", "in0");

  // Check the requirement updates.shape = indices.shape + ref.shape[1:].
  INFER_ERROR("Shapes must be equal rank, but are 1 and 0", op, "[2];[];[2]");
}

TEST(StateOpsTest, TemporaryVariable_ShapeFn) {
  ShapeInferenceTestOp op("TemporaryVariable");
  TensorShape shape({1, 2, 3});
  TF_ASSERT_OK(NodeDefBuilder("test", "TemporaryVariable")
                   .Attr("shape", shape)
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "[1,2,3]");
}

TEST(StateOpsTest, Variable_ShapeFn) {
  ShapeInferenceTestOp op("Variable");

  // Unknown rank.
  TF_ASSERT_OK(NodeDefBuilder("test", "Variable")
                   .Attr("shape", PartialTensorShape())
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "?");

  // For historical reasons an empty TensorShapeProto can be either an unknown
  // rank or a scalar, so the shape function conservatively says "unknown"
  TF_ASSERT_OK(NodeDefBuilder("test", "Variable")
                   .Attr("shape", TensorShape({}))
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "?");

  // Specified shape.
  TF_ASSERT_OK(NodeDefBuilder("test", "Variable")
                   .Attr("shape", TensorShape({1, 2, 3}))
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "[1,2,3]");
}

TEST(StateOpsTest, VariableV2_ShapeFn) {
  ShapeInferenceTestOp op("VariableV2");

  // Unknown rank.
  TF_ASSERT_OK(NodeDefBuilder("test", "VariableV2")
                   .Attr("shape", PartialTensorShape())
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "?");

  // Scalar shape.
  TF_ASSERT_OK(NodeDefBuilder("test", "VariableV2")
                   .Attr("shape", TensorShape({}))
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "[]");

  // Specified shape.
  TF_ASSERT_OK(NodeDefBuilder("test", "VariableV2")
                   .Attr("shape", TensorShape({1, 2, 3}))
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "[1,2,3]");
}
}  // end namespace tensorflow
