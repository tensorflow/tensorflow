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
  auto set_n = [&op](int input_count, int output_count) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    for (int i = 0; i < input_count; ++i) {
      src_list.emplace_back("a", 0, DT_FLOAT);
    }
    TF_ASSERT_OK(NodeDefBuilder("test", "RemoteFusedGraphExecute")
                     .Input(src_list)
                     .Attr("M", input_count)
                     .Attr("N", output_count)
                     .Attr("T", DT_FLOAT)
                     .Attr("U", DT_FLOAT)
                     .Finalize(&op.node_def));
  };
  set_n(4, 2);
  INFER_OK(op, "?;?;?;?", "?;?");  // output rank unknown
  // TODO(satok): Implement shape inference and do its test here
}

}  // namespace tensorflow
