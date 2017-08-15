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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(DataFlowOpsTest, LookupTableFind) {
  ShapeInferenceTestOp op("LookupTableFind");
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "[2];[];[]", "?");
  INFER_OK(op, "[?];[1,2,3];[]", "?");

  // Last input must be a scalar or vector
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[?];[1,2,3];[1,2]");
}

TEST(DataFlowOpsTest, LookupTableInsert) {
  ShapeInferenceTestOp op("LookupTableInsert");
  INFER_OK(op, "?;?;?", "");
  INFER_OK(op, "[2];[];[]", "");

  // Dim 0 (table handle) must be a vector.
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];[1,2,3];[]");
  INFER_OK(op, "[2];[1,?,3];[?,2,?]", "");
}

TEST(DataFlowOpsTest, LookupTableSize) {
  ShapeInferenceTestOp op("LookupTableSize");
  // Always scalar output.
  INFER_OK(op, "?", "[]");
  INFER_OK(op, "[2]", "[]");

  // Dim 0 (table handle) must be a vector of length 2
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[]");
  INFER_ERROR("Dimension must be 2 but is 3", op, "[3]");
}

TEST(DataFlowOpsTest, LookupTableExport) {
  ShapeInferenceTestOp op("LookupTableExport");
  // Always one vector output and one unknown size output.
  INFER_OK(op, "?", "[?];?");
  INFER_OK(op, "[2]", "[?];?");

  // Dim 0 (table handle) must be a vector.
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[]");
}

TEST(DataFlowOpsTest, InitializeTable) {
  ShapeInferenceTestOp op("InitializeTable");
  // Always no output.
  INFER_OK(op, "?;?;?", "");

  // Dim 0 (table handle) must be a vector.
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];[];[]");

  // Dims 1 and 2 (keys and values) are the same size and must be vectors.
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "?;[1];[2]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[2];[1,2];[1,2]");
}

TEST(DataFlowOpsTest, InitializeTableFromTextFile) {
  ShapeInferenceTestOp op("InitializeTableFromTextFile");
  // Always no output.
  INFER_OK(op, "?;?", "");

  // Dim 1 (filename) must be scalar, dim 0 vector
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];[]");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[2];[1]");
}

TEST(DataFlowOpsTest, DynamicPartition) {
  ShapeInferenceTestOp op("DynamicPartition");
  TF_ASSERT_OK(NodeDefBuilder("test", "DynamicPartition")
                   .Input("data", 0, DT_FLOAT_REF)
                   .Input("indices", 0, DT_INT32)
                   .Attr("num_partitions", 4)
                   .Finalize(&op.node_def));

  // Unknown rank for indices, so unknown shape.
  INFER_OK(op, "?;?", "?;?;?;?");

  // 3 dimensional data, 2 dimensional indices.
  INFER_OK(op, "[3,4,5];[3,4]", "[?,d0_2];[?,d0_2];[?,d0_2];[?,d0_2]");

  TF_ASSERT_OK(NodeDefBuilder("test", "DynamicPartition")
                   .Input("data", 0, DT_FLOAT)
                   .Input("indices", 0, DT_INT32)
                   .Attr("num_partitions", 2)
                   .Finalize(&op.node_def));

  // Suffix after matching prefix is copied over.
  INFER_OK(op, "[3,4,5,6];[3,4]", "[?,d0_2,d0_3];[?,d0_2,d0_3]");

  // Does not start with proper prefix
  INFER_ERROR("Dimensions must be equal, but are 4 and 100", op,
              "[3,4,5];[3,100]");
}

TEST(DataFlowOpsTest, DynamicStitch) {
  ShapeInferenceTestOp op("DynamicStitch");
  TF_ASSERT_OK(
      NodeDefBuilder("test", "DynamicStitch")
          .Input({{"indices", 0, DT_INT32}, {"indices_2", 1, DT_INT32}})
          .Input({{"data", 0, DT_FLOAT}, {"data_2", 1, DT_FLOAT}})
          .Attr("N", 2)
          .Finalize(&op.node_def));

  INFER_OK(op, "[2,3];[5,6];[2,3,4,5];[5,6,4,5]", "[?,d2_2,d2_3]");

  // Bad prefix for the second data input.
  INFER_ERROR("Dimensions must be equal, but are 10 and 5", op,
              "[2,3];[5,6];[2,3,4,5];[10,11,4,5]");

  // Inconsistent suffix dimensions
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 4 and 13", op,
              "[2,3];[5,6];[2,3,4,5];[5,6,13,14]");
}

TEST(DataFlowOpsTest, ParallelDynamicStitch) {
  ShapeInferenceTestOp op("ParallelDynamicStitch");
  TF_ASSERT_OK(
      NodeDefBuilder("test", "ParallelDynamicStitch")
          .Input({{"indices", 0, DT_INT32}, {"indices_2", 1, DT_INT32}})
          .Input({{"data", 0, DT_FLOAT}, {"data_2", 1, DT_FLOAT}})
          .Attr("N", 2)
          .Finalize(&op.node_def));

  INFER_OK(op, "[2,3];[5,6];[2,3,4,5];[5,6,4,5]", "[?,d2_2,d2_3]");

  // Bad prefix for the second data input.
  INFER_ERROR("Dimensions must be equal, but are 10 and 5", op,
              "[2,3];[5,6];[2,3,4,5];[10,11,4,5]");

  // Inconsistent suffix dimensions
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 4 and 13", op,
              "[2,3];[5,6];[2,3,4,5];[5,6,13,14]");
}

TEST(DataFlowOpsTest, TensorArrayV3) {
  ShapeInferenceTestOp op("TensorArrayV3");
  TF_ASSERT_OK(NodeDefBuilder("test", "TensorArrayV3")
                   .Input({"size", 0, DT_INT32})
                   .Attr("dtype", DT_FLOAT)
                   .Finalize(&op.node_def));

  INFER_OK(op, "[]", "[2];[]");
  INFER_OK(op, "?", "[2];[]");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[2]");
}

TEST(DataFlowOpsTest, QueueDequeueV2ShapeFn) {
  ShapeInferenceTestOp op("QueueDequeueV2");
  TF_ASSERT_OK(NodeDefBuilder("test", op.name)
                   .Input("handle", 0, DT_RESOURCE)
                   .Attr("component_types", {DT_FLOAT, DT_INT32})
                   .Finalize(&op.node_def));

  INFER_OK(op, "?", "?;?");

  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  INFER_OK(op, "?", "?;?");

  // Wrong number of shapes provided by handle.
  shapes_and_types.emplace_back("[1,?,3]", DT_FLOAT);
  INFER_OK(op, "?", "?;?");

  // Correct number of shapes provided by handle.
  shapes_and_types.emplace_back("[?,2]", DT_FLOAT);
  INFER_OK(op, "?", "[1,?,3];[?,2]");
}

TEST(DataFlowOpsTest, QueueDequeueManyV2ShapeFn) {
  ShapeInferenceTestOp op("QueueDequeueManyV2");
  TF_ASSERT_OK(NodeDefBuilder("test", op.name)
                   .Input("handle", 0, DT_RESOURCE)
                   .Input("n", 0, DT_INT32)
                   .Attr("component_types", {DT_FLOAT, DT_INT32})
                   .Finalize(&op.node_def));

  ////////////////////////////
  // Input n is not a constant.
  INFER_OK(op, "?;?", "?;?");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  // Wrong number of shapes provided by handle.
  shapes_and_types.emplace_back("[1,?,3]", DT_FLOAT);
  INFER_OK(op, "?;?", "?;?");
  // Correct number of shapes provided by handle.
  shapes_and_types.emplace_back("[?,2]", DT_FLOAT);
  INFER_OK(op, "?;?", "[?,1,?,3];[?,?,2]");

  ////////////////////////////
  // Input n is a constant. (set up test and repeat the cases from above).
  Tensor n_tensor = test::AsScalar(12);
  op.input_tensors.push_back(nullptr);
  op.input_tensors.push_back(&n_tensor);
  op.input_resource_handle_shapes_and_types.clear();
  shapes_and_types.clear();

  INFER_OK(op, "?;?", "?;?");
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  // Wrong number of shapes provided by handle.
  shapes_and_types.emplace_back("[1,?,3]", DT_FLOAT);
  INFER_OK(op, "?;?", "?;?");
  // Correct number of shapes provided by handle.
  shapes_and_types.emplace_back("[?,2]", DT_FLOAT);
  INFER_OK(op, "?;?", "[12,1,?,3];[12,?,2]");

  n_tensor = test::AsScalar<int32>(-1);  // invalid value of n.
  INFER_ERROR("must be >= 0", op, "?;?");
}

TEST(DataFlowOpsTest, QueueDequeueUpToV2ShapeFn) {
  // Results are the same regardless of what value is passed for n.
  for (int pass = 0; pass < 2; ++pass) {
    ShapeInferenceTestOp op("QueueDequeueUpToV2");
    TF_ASSERT_OK(NodeDefBuilder("test", op.name)
                     .Input("handle", 0, DT_RESOURCE)
                     .Input("n", 0, DT_INT32)
                     .Attr("component_types", {DT_FLOAT, DT_INT32})
                     .Finalize(&op.node_def));

    Tensor n_tensor = test::AsScalar(12);
    if (pass == 1) {
      // Second pass, pass value of <n> as a constant.
      op.input_tensors.push_back(nullptr);
      op.input_tensors.push_back(&n_tensor);
    }

    INFER_OK(op, "?;?", "?;?");
    std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types;
    op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
    op.input_resource_handle_shapes_and_types.push_back(nullptr);
    // Wrong number of shapes provided by handle.
    shapes_and_types.emplace_back("[1,?,3]", DT_FLOAT);
    INFER_OK(op, "?;?", "?;?");
    // Correct number of shapes provided by handle.
    shapes_and_types.emplace_back("[?,2]", DT_FLOAT);
    INFER_OK(op, "?;?", "[?,1,?,3];[?,?,2]");
  }
}

}  // end namespace tensorflow
