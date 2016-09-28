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

TEST(CudnnRNNOpsTest, ParamsSize_ShapeFn) {
  ShapeInferenceTestOp op("CudnnRNNParamsSize");
  INFER_OK(op, "[1];[1];[1]", "[1]");
}

TEST(CudnnRNNOpsTest, ForwardLstm_ShapeFn) {
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
    return strings::StrCat("[", str_util::Join(v, ","), "]");
  };
  string input_shapes_desc = strings::StrCat(
      shape_to_str(input_shape), ";", shape_to_str(input_h_shape), ";",
      shape_to_str(input_h_shape), ";", "[?]");
  string output_shapes_desc = "[d0_0,d0_1,d1_2];in1;in1;?";
  INFER_OK(op, input_shapes_desc, output_shapes_desc);
}

}  // end namespace tensorflow
