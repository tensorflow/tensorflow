/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(RemoteFusedGraphOpsTest, RemoteFusedGraphExecute_ShapeFn) {
  ShapeInferenceTestOp op("RemoteFusedGraphExecute");
  auto set_n = [&op](int input1_count, int input2_count, int output_count) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    DataTypeVector input_types;
    for (int i = 0; i < input1_count; ++i) {
      src_list.emplace_back("a", 0, DT_FLOAT);
      input_types.emplace_back(DT_FLOAT);
    }
    for (int i = 0; i < input2_count; ++i) {
      src_list.emplace_back("b", 0, DT_INT32);
      input_types.emplace_back(DT_INT32);
    }
    DataTypeVector output_types;
    for (int i = 0; i < output_count; ++i) {
      output_types.emplace_back(DT_FLOAT);
    }
    NodeDefBuilder builder = NodeDefBuilder("test", "RemoteFusedGraphExecute")
                                 .Input(src_list)
                                 .Attr("Tinputs", input_types)
                                 .Attr("Toutputs", output_types);
    TF_ASSERT_OK(builder.Finalize(&op.node_def));
  };
  set_n(4, 0, 2);
  INFER_OK(op, "?;?;?;?", "?;?");  // output rank unknown

  set_n(4, 3, 3);
  INFER_OK(op, "?;?;?;?;?;?;?", "?;?;?");  // output rank unknown

  // TODO(satok): Implement shape inference and do its test here
}

}  // namespace tensorflow
