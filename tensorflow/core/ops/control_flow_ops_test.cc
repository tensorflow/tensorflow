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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ControlFlowOpsTest, Merge_ShapeFn) {
  ShapeInferenceTestOp op("Merge");

  int n = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.reserve(n);
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
  TF_ASSERT_OK(NodeDefBuilder("test", "Merge")
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));

  // The second output should always be scalar.
  // The first output should be unknown if any of the inputs are unknown, or
  // if two inputs disagree about rank.
  INFER_OK(op, "?;?;?", "?;[]");
  INFER_OK(op, "[2,1];?;[2,1]", "?;[]");
  INFER_OK(op, "[2,1];[2,1];?", "?;[]");
  INFER_OK(op, "[2,1];[2,1];[3,1,2]", "?;[]");
  // If inputs on rank, but disagree on specific dimensions, those dimensions
  // should be unknown.
  INFER_OK(op, "[2,1];[2,1];[3,1]", "[?,d0_1];[]");
  INFER_OK(op, "[2,1];[2,2];[3,1]", "[?,?];[]");
  // Otherwise, all inputs agree and we return the first input.
  INFER_OK(op, "[2,1];[2,1];[2,1]", "in0;[]");
}

TEST(ControlFlowOpsTest, RefSelect_ShapeFn) {
  ShapeInferenceTestOp op("RefSelect");

  int n = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  src_list.reserve(n);
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 1, DT_FLOAT_REF);
  TF_ASSERT_OK(NodeDefBuilder("test", "RefSelect")
                   .Input("index", 0, DT_INT32)
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));

  // The first argument should be scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[2];?;?;?");

  // If any inputs aren't fully defined, we return an unknown shape.
  INFER_OK(op, "?;?;?;?", "?");
  INFER_OK(op, "[];?;?;?", "?");
  INFER_OK(op, "[];[1,2,3];?;?", "?");
  INFER_OK(op, "[];[1,2,3];[1,2,?];[1,2,3]", "?");
  // If inputs disagree on rank or dimension, we return an unknown shape.
  INFER_OK(op, "[];[1,2,3];[1,2];[1,2,3]", "?");
  INFER_OK(op, "[];[1,2,3];[1,2,4];[1,2,3]", "?");
  // Otherwise, all inputs agree and we return the first input.
  INFER_OK(op, "[];[1,2,3];[1,2,3];[1,2,3]", "in1");
}

}  // end namespace tensorflow
