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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static string JoinedCopies(const string& s, int copies) {
  string res;
  for (int i = 0; i < copies; ++i) {
    absl::StrAppend(&res, i > 0 ? ";" : "", s);
  }
  return res;
}

TEST(CudnnRNNOpsTest, ParamsSize_ShapeFn) {
  ShapeInferenceTestOp op("CudnnRNNParamsSize");
  INFER_OK(op, "[];[];[]", "[1]");
  INFER_OK(op, "?;[];[]", "[1]");
  INFER_OK(op, "[];?;[]", "[1]");
  INFER_OK(op, "[];[];?", "[1]");
  INFER_OK(op, "[];?;?", "[1]");
  INFER_OK(op, "?;?;?", "[1]");

  INFER_ERROR("Shape must be rank 0 ", op, "[1,2];?;[]");
  INFER_ERROR("Shape must be rank 0 ", op, "?;[2];[]");
  INFER_ERROR("Shape must be rank 0 ", op, "?;?;[1]");
}

TEST(CudnnRNNOpsTest, ForwardLstm_ShapeFn) {
  int seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {seq_length, batch_size,
                                   num_units * dir_count};
  auto shape_to_str = [](const std::vector<int>& v) {
    return absl::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_h_shape), ";", "[?]");
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in1;?";

  ShapeInferenceTestOp op("CudnnRNN");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNN")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Attr("rnn_mode", "lstm")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[?,?,?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[?,?,?];[?]");
  // Disabled because the kernel does not check shape of input_c.
  // INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[]");
}

TEST(CudnnRNNOpsTest, ForwardV2Lstm_ShapeFn) {
  int seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {seq_length, batch_size,
                                   num_units * dir_count};
  auto shape_to_str = [](const std::vector<int>& v) {
    return absl::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_h_shape), ";", "[?]");
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in1;?;?";

  ShapeInferenceTestOp op("CudnnRNNV2");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNNV2")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Attr("rnn_mode", "lstm")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[?,?,?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[?,?,?];[?]");
  // Disabled because the kernel does not check shape of input_c.
  // INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[]");
}

TEST(CudnnRNNOpsTest, ForwardV3Lstm_ShapeFn) {
  int max_seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {max_seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> input_c_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {max_seq_length, batch_size,
                                   num_units * dir_count};
  std::vector<int> seq_lengths_shape = {batch_size};
  auto shape_to_str = [](const std::vector<int>& v) {
    return absl::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_c_shape), ";", "[?]", ";",
      shape_to_str(seq_lengths_shape));
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in2;?;?";

  ShapeInferenceTestOp op("CudnnRNNV3");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNNV3")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Input({"sequence_lengths", 0, DT_INT32})
                   .Attr("rnn_mode", "lstm")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[?,?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[?,?,?];[];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[?,?,?];[?];[]");
}

TEST(CudnnRNNOpsTest, ForwardV3Gru) {
  int max_seq_length = 2;
  int batch_size = 3;
  int num_units = 4;
  int num_layers = 5;
  int dir_count = 1;
  std::vector<int> input_shape = {max_seq_length, batch_size, num_units};
  std::vector<int> input_h_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> input_c_shape = {num_layers * dir_count, batch_size,
                                    num_units};
  std::vector<int> output_shape = {max_seq_length, batch_size,
                                   num_units * dir_count};
  std::vector<int> seq_lengths_shape = {batch_size};
  auto shape_to_str = [](const std::vector<int>& v) {
    return absl::StrCat("[", absl::StrJoin(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_c_shape), ";", "[?]", ";",
      shape_to_str(seq_lengths_shape));
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;[];?;?";

  ShapeInferenceTestOp op("CudnnRNNV3");
  TF_ASSERT_OK(NodeDefBuilder("test", "CudnnRNNV3")
                   .Input({"input", 0, DT_FLOAT})
                   .Input({"input_h", 0, DT_FLOAT})
                   .Input({"input_c", 0, DT_FLOAT})
                   .Input({"params", 0, DT_FLOAT})
                   .Input({"sequence_lengths", 0, DT_INT32})
                   .Attr("rnn_mode", "gru")
                   .Attr("input_mode", "auto_select")
                   .Attr("direction", "unidirectional")
                   .Finalize(&op.node_def));
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
  INFER_ERROR("Shape must be rank 3 ", op, "[];[?,?,?];[];[?];[?]");
  INFER_ERROR("Shape must be rank 3 ", op, "[?,?,?];[];[];[?];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[];[];[?]");
  INFER_ERROR("Shape must be rank 1 ", op, "[?,?,?];[?,?,?];[];[?];[]");
}

TEST(CudnnRNNOpsTest, LSTMBlockCell_ShapeFn) {
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

TEST(CudnnRNNOpsTest, BlockLSTM_ShapeFn) {
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

TEST(CudnnRNNOpsTest, BlockLSTMV2_ShapeFn) {
  ShapeInferenceTestOp op("BlockLSTMV2");

  TF_ASSERT_OK(NodeDefBuilder("test", "BlockLSTMV2")
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

}  // end namespace tensorflow
