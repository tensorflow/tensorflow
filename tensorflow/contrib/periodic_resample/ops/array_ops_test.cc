/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ArrayOpsTest, PeriodicResample_ShapeFn) {
  ShapeInferenceTestOp op("PeriodicResample");
  // Case 1: output shape can be fully inferreed.
  PartialTensorShape shape({4, 4, -1});
  TensorShapeProto shape_proto;
  shape.AsProto(&shape_proto);

  TF_ASSERT_OK(NodeDefBuilder("test", "PeriodicResample")
                   .Input({"values", 0, DT_INT32})
                   .Attr("shape", shape_proto)
                   .Finalize(&op.node_def));
  INFER_OK(op, "[2,2,4]", "[4,4,1]");
  // Case 2: output shape can not be inferred - report desired shape.
  INFER_OK(op, "[2,2,?]", "[4,4,?]");
}

}  // end namespace tensorflow
