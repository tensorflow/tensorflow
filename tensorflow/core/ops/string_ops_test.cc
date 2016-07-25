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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(StringOpsTest, ReduceJoin_ShapeFn) {
  ShapeInferenceTestOp op("ReduceJoin");

#if 0
  INFER_OK(op, "?;?;?;?", "");
  INFER_OK(op, "[];[2];?;?", "");

  // Filename must be scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[?];?;?;?");

  // tensor_names must be vector matching number data elements (2 in this test).
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[];[2,3];?;?");
  INFER_ERROR("Dimension must be 2 but is 3", op, "[];[3];?;?");
#endif
}

TEST(StringOpsTest, StringJoin_ShapeFn) {
  ShapeInferenceTestOp op("StringJoin");
  int n = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_STRING);
  TF_CHECK_OK(NodeDefBuilder("test", "StringJoin")
                  .Input(src_list)
                  .Attr("n", n)
                  .Finalize(&op.node_def));

  // If all inputs are scalar, return a scalar.
  INFER_OK(op, "[];[];[]", "[]");

  // If one input is unknown, but rest scalar, return unknown.  Technically this
  // could return in1, but we don't optimize this case yet.
  INFER_OK(op, "[];?;[]", "?");

  // Inputs that are non-scalar are merged to produce the output.
  INFER_OK(op, "[1,?];[];[?,2]", "[d0_0,d2_1]");
  INFER_OK(op, "[1,?];?;[?,2]", "[d0_0,d2_1]");
  INFER_ERROR("must be equal", op, "[1,2];[];[?,3]");
}

}  // end namespace tensorflow
