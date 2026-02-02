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

TEST(ParsingOpsTest, DecodeRaw_ShapeFn) {
  ShapeInferenceTestOp op("DecodeRaw");

  // Output is input + an unknown dim.
  INFER_OK(op, "?", "?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,d0_2,?]");
}

TEST(ParsingOpsTest, DecodeCSV_ShapeFn) {
  ShapeInferenceTestOp op("DecodeCSV");
  auto set_n_outputs = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    std::vector<DataType> out_types;
    for (int i = 0; i < n; ++i) {
      src_list.emplace_back("b", 0, DT_FLOAT);
      out_types.push_back(DT_FLOAT);
    }
    TF_CHECK_OK(NodeDefBuilder("test", "DecodeCSV")
                    .Input("a", 0, DT_STRING)
                    .Input(src_list)
                    .Attr("OUT_TYPE", out_types)
                    .Finalize(&op.node_def));
  };

  // Output is always n copies of input 0.
  set_n_outputs(2);
  INFER_OK(op, "?;?;?", "in0;in0");
  INFER_OK(op, "[1,2,?,4];?;?", "in0;in0");
  INFER_OK(op, "[1,2,?,4];[?];[?]", "in0;in0");

  // Check errors in the record_defaults inputs.
  INFER_ERROR("must be rank 1", op, "?;?;[]");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("Shape of a default must be", op, "?;?;[2]");
  INFER_ERROR("Shape of a default must be", op, "?;[2];?");
}

}  // end namespace tensorflow
