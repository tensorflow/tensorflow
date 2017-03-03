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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(FunctionalOpsTest, SymbolicGradient_ShapeFn) {
  ShapeInferenceTestOp op("SymbolicGradient");
  int num_inputs = 4;
  int num_outputs = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  std::vector<DataType> in_type_list;
  std::vector<DataType> out_type_list;
  for (int i = 0; i < num_inputs; ++i) {
    in_type_list.emplace_back(DT_FLOAT);
    src_list.emplace_back("a", 0, DT_FLOAT);
  }
  for (int i = 0; i < num_outputs; ++i) {
    out_type_list.emplace_back(DT_FLOAT);
  }
  TF_ASSERT_OK(NodeDefBuilder("test", "SymbolicGradient")
                   .Input(src_list)
                   .Attr("Tin", in_type_list)
                   .Attr("Tout", out_type_list)
                   .Finalize(&op.node_def));

  // Inputs transferred to outputs.
  INFER_OK(op, "?;?;?;?", "in0;in1;in2");
  INFER_OK(op, "[];[2];?;?", "in0;in1;in2");
}

}  // end namespace tensorflow
