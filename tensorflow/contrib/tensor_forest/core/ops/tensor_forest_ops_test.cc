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

// TODO(cwhipkey): iwyu
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(TrainingOpsTest, UpdateFertileSlots_ShapeFn) {
  ShapeInferenceTestOp op("UpdateFertileSlots");
  INFER_OK(op, "?;?;?;?;?;?;?;?", "[2,?];[2,?];[?];[?]");
}

TEST(TrainingOpsTest, ScatterAddNdim_ShapeFn) {
  ShapeInferenceTestOp op("ScatterAddNdim");
  INFER_OK(op, "?;?;?", "");
}

TEST(TrainingOpsTest, GrowTree_ShapeFn) {
  ShapeInferenceTestOp op("GrowTree");
  INFER_OK(op, "?;?;?;?;?;?", "[?];[?,2];[?];[1]");
}

TEST(TrainingOpsTest, FinishedNodes_ShapeFn) {
  ShapeInferenceTestOp op("FinishedNodes");
  INFER_OK(op, "?;?;?;?;?;?;?;?", "[?];[?]");
}

TEST(TrainingOpsTest, BestSplits_ShapeFn) {
  ShapeInferenceTestOp op("BestSplits");
  INFER_OK(op, "?;?;?;?;?;?", "[?]");
  INFER_OK(op, "[?];?;?;?;?;?", "[d0_0]");
  INFER_OK(op, "[1];?;?;?;?;?", "[d0_0]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2];?;?;?;?;?");
}

TEST(TrainingOpsTest, SampleInputs_ShapeFn) {
  ShapeInferenceTestOp op("SampleInputs");

  // input[7].dim(1) determines dims in the output.
  INFER_OK(op, "?;?;?;?;?;?;?;?;?", "[?];[?,?];[?,?]");
  INFER_OK(op, "?;?;?;?;?;?;?;[?,?];?", "[?];[?,d7_1];[?,d7_1]");
  INFER_OK(op, "?;?;?;?;?;?;?;[1,2];?", "[?];[?,d7_1];[?,d7_1]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op,
              "?;?;?;?;?;?;?;[1,2,3];?");
}

TEST(TrainingOpsTest, CountExtremelyRandomStats_ShapeFn) {
  ShapeInferenceTestOp op("CountExtremelyRandomStats");
  TF_ASSERT_OK(NodeDefBuilder("test", "CountExtremelyRandomStats")
                   .Input("input_data", 0, DT_FLOAT)
                   .Input("sparse_input_indices", 1, DT_INT64)
                   .Input("sparse_input_values", 2, DT_FLOAT)
                   .Input("sparse_input_shape", 3, DT_INT64)
                   .Input("input_spec", 4, DT_INT32)
                   .Input("input_labels", 5, DT_FLOAT)
                   .Input("input_weights", 6, DT_FLOAT)
                   .Input("tree", 7, DT_INT32)
                   .Input("tree_thresholds", 8, DT_FLOAT)
                   .Input("node_to_accumulator", 9, DT_INT32)
                   .Input("candidate_split_features", 10, DT_INT32)
                   .Input("candidate_split_thresholds", 11, DT_FLOAT)
                   .Input("birth_epochs", 12, DT_INT32)
                   .Input("current_epoch", 13, DT_INT32)
                   .Attr("num_classes", 10)
                   .Attr("regression", false)
                   .Finalize(&op.node_def));

  // num_points = 2, num_nodes = 4, regression = false, num_classes = 10
  // num_nodes = 4
  INFER_OK(op, "[2,3];?;?;?;?;?;?;[4];?;?;?;?;?;?",
           "[d7_0,10];[d7_0,10];[?,3];[?];[0];[?,2];[?];[0];[d0_0]");

  TF_ASSERT_OK(NodeDefBuilder("test", "CountExtremelyRandomStats")
                   .Input("input_data", 0, DT_FLOAT)
                   .Input("sparse_input_indices", 1, DT_INT64)
                   .Input("sparse_input_values", 2, DT_FLOAT)
                   .Input("sparse_input_shape", 3, DT_INT64)
                   .Input("input_spec", 4, DT_INT32)
                   .Input("input_labels", 5, DT_FLOAT)
                   .Input("input_weights", 6, DT_FLOAT)
                   .Input("tree", 7, DT_INT32)
                   .Input("tree_thresholds", 8, DT_FLOAT)
                   .Input("node_to_accumulator", 9, DT_INT32)
                   .Input("candidate_split_features", 10, DT_INT32)
                   .Input("candidate_split_thresholds", 11, DT_FLOAT)
                   .Input("birth_epochs", 12, DT_INT32)
                   .Input("current_epoch", 13, DT_INT32)
                   .Attr("num_classes", 10)
                   .Attr("regression", true)
                   .Finalize(&op.node_def));

  // num_points = 2, num_nodes = 4, regression = false, num_classes = 10
  // num_nodes = 4
  INFER_OK(
      op, "[2,3];?;?;?;?;?;?;[4];?;?;?;?;?;?",
      "[d7_0,10];[d7_0,10];[?,2];[?,10];[?,10];[?,1];[?,10];[?,10];[d0_0]");

  // Sparse shape known and > 1, so num_points is unknown
  INFER_OK(op, "[2,3];?;?;[10,11];?;?;?;[4];?;?;?;?;?;?",
           "[d7_0,10];[d7_0,10];[?,2];[?,10];[?,10];[?,1];[?,10];[?,10];[?]");
}

TEST(TrainingOpsTest, TreePredictions_ShapeFn) {
  ShapeInferenceTestOp op("TreePredictions");
  TF_ASSERT_OK(NodeDefBuilder("test", "TreePredictions")
                   .Input("a", 0, DT_FLOAT)
                   .Input("b", 1, DT_INT64)
                   .Input("c", 2, DT_FLOAT)
                   .Input("d", 3, DT_INT64)
                   .Input("e", 4, DT_INT32)
                   .Input("f", 5, DT_INT32)
                   .Input("g", 6, DT_FLOAT)
                   .Input("h", 7, DT_FLOAT)
                   .Attr("valid_leaf_threshold", 0.5)
                   .Finalize(&op.node_def));

  // num_points = 2, num_classes = 10, sparse shape not known
  INFER_OK(op, "[2,3];?;?;?;?;?;?;[1,10]", "[d0_0,9]");

  // num_points = 2, num_classes = 10, sparse shape rank known and > 1
  INFER_OK(op, "[2,3];?;?;[10,11];?;?;?;[1,10]", "[?,9]");
}

}  // namespace tensorflow
