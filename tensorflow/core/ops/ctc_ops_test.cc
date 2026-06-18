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

TEST(CtcOpsTest, CTCLoss_ShapeFn) {
  ShapeInferenceTestOp op("CTCLoss");

  // Inputs are inputs, labels_indices, labels_values, and sequence_length.

  // Rank checks
  INFER_ERROR("must be rank 3", op, "[];?;?;?");  // inputs
  INFER_ERROR("must be rank 2", op, "?;[];?;?");  // labels_indices
  INFER_ERROR("must be rank 1", op, "?;?;[];?");  // labels_values
  INFER_ERROR("must be rank 1", op, "?;?;?;[]");  // sequence_length

  // labels_indices.dim(0) and labels_values.dim(0) must match.
  INFER_ERROR("must be equal", op, "?;[1,?];[2];?");

  // batch_size comes from inputs.dim(1) merged with sequence_length.dim(0).
  // This becomes the dimension of the first out, and replaced inputs.dim(1) to
  // become the second out.
  INFER_OK(op, "[?,?,?];?;?;[?]", "[d0_1|d3_0];[d0_0,d0_1|d3_0,d0_2]");
  INFER_OK(op, "[?,1,?];?;?;[1]", "[d0_1|d3_0];[d0_0,d0_1|d3_0,d0_2]");
  INFER_OK(op, "[?,?,?];?;?;[1]", "[d3_0];[d0_0,d3_0,d0_2]");
  INFER_OK(op, "[?,1,?];?;?;[?]", "[d0_1];[d0_0,d0_1,d0_2]");
  INFER_ERROR("must be equal", op, "[?,1,?];?;?;[2]");
}

TEST(CtcOpsTest, CTCGreedyDecoder_ShapeFn) {
  ShapeInferenceTestOp op("CTCGreedyDecoder");

  // Inputs are inputs and sequence_length.

  // Rank checks
  INFER_ERROR("must be rank 3", op, "[];?");  // inputs
  INFER_ERROR("must be rank 1", op, "?;[]");  // sequence_length

  // batch_size comes from inputs.dim(1) merged with sequence_length.dim(0).
  // This becomes outputs[3].dim(0).
  INFER_OK(op, "[?,?,?];[?]", "[?,2];[?];[2];[d0_1|d1_0,1]");
  INFER_OK(op, "[?,1,?];[1]", "[?,2];[?];[2];[d0_1|d1_0,1]");
  INFER_OK(op, "[?,?,?];[1]", "[?,2];[?];[2];[d1_0,1]");
  INFER_OK(op, "[?,1,?];[?]", "[?,2];[?];[2];[d0_1,1]");
  INFER_ERROR("must be equal", op, "[?,1,?];[2]");
}

TEST(CtcOpsTest, CTCBeamSearchDecoder_ShapeFn) {
  ShapeInferenceTestOp op("CTCBeamSearchDecoder");
  auto set_top_paths = [&op](int top_paths) {
    TF_ASSERT_OK(NodeDefBuilder("test", "CTCBeamSearchDecoder")
                     .Input({"a", 0, DT_FLOAT})
                     .Input({"b", 0, DT_INT32})
                     .Attr("top_paths", top_paths)
                     .Finalize(&op.node_def));
  };
  set_top_paths(1);

  // Inputs are inputs and sequence_length.

  // Rank checks
  INFER_ERROR("must be rank 3", op, "[];?");  // inputs
  INFER_ERROR("must be rank 1", op, "?;[]");  // sequence_length

  // batch_size comes from inputs.dim(1) merged with sequence_length.dim(0).
  // This becomes dim(0) of the final output shape.
  INFER_OK(op, "[?,?,?];[?]", "[?,2];[?];[2];[d0_1|d1_0,1]");
  INFER_OK(op, "[?,1,?];[1]", "[?,2];[?];[2];[d0_1|d1_0,1]");
  INFER_OK(op, "[?,?,?];[1]", "[?,2];[?];[2];[d1_0,1]");
  INFER_OK(op, "[?,1,?];[?]", "[?,2];[?];[2];[d0_1,1]");
  INFER_ERROR("must be equal", op, "[?,1,?];[2]");

  // test higher top_paths value. Compared to top_paths=1, each of first 3 dims
  // is doubled, and final shape.dim(1) becomes 2.
  set_top_paths(2);
  INFER_OK(op, "?;?", "[?,2];[?,2];[?];[?];[2];[2];[?,2]");
}

}  // end namespace tensorflow
