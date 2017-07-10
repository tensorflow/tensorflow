// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ModelOpsTest, CreateTreeVariable_ShapeFn) {
  ShapeInferenceTestOp op("CreateTreeVariable");
  INFER_OK(op, "[1];[1]", "");
}

TEST(ModelOpsTest, TreeSerialize_ShapeFn) {
  ShapeInferenceTestOp op("TreeSerialize");
  INFER_OK(op, "[1]", "[]");
}

TEST(ModelOpsTest, TreeDeserialize_ShapeFn) {
  ShapeInferenceTestOp op("TreeDeserialize");
  INFER_OK(op, "[1];[1]", "");
}

TEST(ModelOpsTest, TreeSize_ShapeFn) {
  ShapeInferenceTestOp op("TreeSize");
  INFER_OK(op, "[1]", "[]");
}

TEST(ModelOpsTest, TreePredictionsV4_ShapeFn) {
  ShapeInferenceTestOp op("TreePredictionsV4");
  TF_ASSERT_OK(NodeDefBuilder("test", "TreePredictionsV4")
                   .Input("a", 0, DT_RESOURCE)
                   .Input("b", 1, DT_FLOAT)
                   .Input("c", 2, DT_INT64)
                   .Input("d", 3, DT_FLOAT)
                   .Input("e", 5, DT_INT64)
                   .Attr("input_spec", "")
                   .Attr("params", "")
                   .Finalize(&op.node_def));

  // num_points = 2, sparse shape not known
  INFER_OK(op, "?;[2,3];?;?;?", "[d1_0,?]");

  // num_points = 2, sparse and dense shape rank known and > 1
  INFER_OK(op, "?;[2,3];?;?;[10,11]", "[d1_0,?]");

  // num_points = 2, sparse shape rank known and > 1
  INFER_OK(op, "?;?;?;?;[10,11]", "[?,?]");
}

TEST(ModelOpsTest, FeatureUsageCounts_ShapeFn) {
  ShapeInferenceTestOp op("FeatureUsageCounts");
  INFER_OK(op, "[1]", "[?]");
}

}  // namespace tensorflow
