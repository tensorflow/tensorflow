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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static string JoinedCopies(const string& s, int copies) {
  string res;
  for (int i = 0; i < copies; ++i) {
    absl::StrAppend(&res, i > 0 ? ";" : "", s);
  }
  return res;
}

TEST(RnnOpsTest, GRUBlockCell_ShapeFn) {
  ShapeInferenceTestOp op("GRUBlockCell");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;[?];?;?;?;?");

  // Output
  INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?,?];[?,?];[?,?]");
  INFER_OK(op, "[?,?];[?,?];?;?;?;?",
           "[d0_0,d1_1];[d0_0,d1_1];[d0_0,d1_1];[d0_0,d1_1]");
}

TEST(RnnOpsTest, GRUBlockCellGrad_ShapeFn) {
  ShapeInferenceTestOp op("GRUBlockCellGrad");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?;?;?;?;?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;[?];?;?;?;?;?;?;?;?");
  INFER_ERROR("must be rank 2", op, "?;?;[?];?;?;?;?;?;?;?");

  // Output
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?", "[?,?];[?,?];[?,?];[?,?]");
  INFER_OK(op, "[?,?];[?,?];[?,?];?;?;?;?;?;?;?",
           "in0;[d0_0,d1_1];[d0_0,d1_1];[d0_0,d2_1]");
}

TEST(RnnOpsTest, LSTMBlockCell_ShapeFn) {
  ShapeInferenceTestOp op("LSTMBlockCell");

  // Last 6 inputs don't affect shape inference.
  string input_suffix = absl::StrCat(";", JoinedCopies("?", 6));

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?" + input_suffix);
  INFER_ERROR("must be rank 2", op, "?;[?]" + input_suffix);

  // Output
  INFER_OK(op, "?;?" + input_suffix, JoinedCopies("[?,?]", 7));
  INFER_OK(op, "[?,?];[?,?]" + input_suffix, JoinedCopies("[d0_0,d1_1]", 7));
}

TEST(RnnOpsTest, LSTMBlockCellGrad_ShapeFn) {
  ShapeInferenceTestOp op("LSTMBlockCellGrad");

  // Last 14 inputs don't affect shape inference.
  string input_suffix = absl::StrCat(";", JoinedCopies("?", 14));

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[?];?" + input_suffix);
  INFER_ERROR("must be rank 2", op, "?;[?]" + input_suffix);

  // Output
  INFER_OK(op, "?;?" + input_suffix, "[?,?];[?,?];[?];[?];[?]");
  INFER_OK(op, "[?,?];[?,?]" + input_suffix,
           "[d0_0,d1_1];[d0_0,?];[d1_1];[d1_1];[d1_1]");
  INFER_OK(op, "[1,2];[3,4]" + input_suffix,
           "[d0_0,d1_1];[d0_0,16];[d1_1];[d1_1];[d1_1]");
}

TEST(RnnOpsTest, BlockLSTM_ShapeFn) {
  ShapeInferenceTestOp op("BlockLSTM");

  TF_ASSERT_OK(NodeDefBuilder("test", "BlockLSTM")
                   .Input({"seq_len_max", 0, DT_INT64})
                   .Input({"x", 0, DT_FLOAT})
                   .Input({"cs_prev", 0, DT_FLOAT})
                   .Input({"h_prev", 0, DT_FLOAT})
                   .Input({"w", 0, DT_FLOAT})
                   .Input({"wci", 0, DT_FLOAT})
                   .Input({"wcf", 0, DT_FLOAT})
                   .Input({"wco", 0, DT_FLOAT})
                   .Input({"b", 0, DT_FLOAT})
                   .Finalize(&op.node_def));

  // Middle inputs don't affect shape inference.
  string infix = ";" + JoinedCopies("?", 6) + ";";

  // Rank checks.
  INFER_ERROR("must be rank 3", op, "?;[?]" + infix + "?");
  INFER_ERROR("must be rank 1", op, "?;?" + infix + "[?,?]");

  // Output
  INFER_OK(op, "?;?" + infix + "?", JoinedCopies("[?,?,?]", 7));
  INFER_OK(op, "?;[?,?,?]" + infix + "?", JoinedCopies("[d1_0,d1_1,?]", 7));
  INFER_OK(op, "?;[?,?,?]" + infix + "[?]", JoinedCopies("[d1_0,d1_1,?]", 7));
  INFER_OK(op, "?;[?,?,?]" + infix + "[20]", JoinedCopies("[d1_0,d1_1,5]", 7));

  // cell_size must be divisible by 4.
  INFER_ERROR("must be evenly divisible", op, "?;?" + infix + "[11]");
}

TEST(RnnOpsTest, BlockLSTMGrad_ShapeFn) {
  ShapeInferenceTestOp op("BlockLSTMGrad");
  TF_ASSERT_OK(NodeDefBuilder("test", "BlockLSTMGrad")
                   .Input({"seq_len_max", 0, DT_INT64})
                   .Input({"x", 0, DT_FLOAT})
                   .Input({"cs_prev", 0, DT_FLOAT})
                   .Input({"h_prev", 0, DT_FLOAT})
                   .Input({"w", 0, DT_FLOAT})
                   .Input({"wci", 0, DT_FLOAT})
                   .Input({"wcf", 0, DT_FLOAT})
                   .Input({"wco", 0, DT_FLOAT})
                   .Input({"b", 0, DT_FLOAT})
                   .Input({"i", 0, DT_FLOAT})
                   .Input({"cs", 0, DT_FLOAT})
                   .Input({"f", 0, DT_FLOAT})
                   .Input({"o", 0, DT_FLOAT})
                   .Input({"ci", 0, DT_FLOAT})
                   .Input({"co", 0, DT_FLOAT})
                   .Input({"h", 0, DT_FLOAT})
                   .Input({"cs_grad", 0, DT_FLOAT})
                   .Input({"h_grad", 0, DT_FLOAT})
                   .Finalize(&op.node_def));

  // Last inputs don't affect shape inference.
  string suffix = ";" + JoinedCopies("?", 9);

  // Rank check for x
  INFER_ERROR("must be rank 3", op, "?;[?];?;?;?;?;?;?;?" + suffix);

  // Rank checks for cs_prev through b.
  INFER_ERROR("must be rank 2", op, "?;?;[1];?;?;?;?;?;?" + suffix);
  INFER_ERROR("must be rank 2", op, "?;?;?;[1];?;?;?;?;?" + suffix);
  INFER_ERROR("must be rank 2", op, "?;?;?;?;[1];?;?;?;?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;[1,?];?;?;?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;?;[1,?];?;?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;?;?;[1,?];?" + suffix);
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;?;?;?;[1,?]" + suffix);

  // Output with all input knowns makes known rank outputs.
  INFER_OK(
      op, JoinedCopies("?", 18),
      "[?,?,?];" + JoinedCopies("[?,?]", 3) + ";" + JoinedCopies("[?]", 4));

  // Output with copies input shapes to output.
  string input = strings::StrCat("?;[?,?,?];", JoinedCopies("[?,?]", 3), ";",
                                 JoinedCopies("[?]", 4), suffix);
  string expected = "in1";
  for (int i = 1; i < 8; ++i) {
    absl::StrAppend(&expected, ";in", i + 1);
  }
  INFER_OK(op, input, expected);
}

}  // namespace tensorflow
