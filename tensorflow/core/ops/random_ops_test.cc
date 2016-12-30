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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(RandomOpsTest, Multinomial_ShapeFn) {
  ShapeInferenceTestOp op("Multinomial");
  op.input_tensors.resize(2);

  INFER_OK(op, "?;?", "[?,?]");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[?];?");
  INFER_OK(op, "[?,?];?", "[d0_0,?]");
  INFER_OK(op, "[2,?];?", "[d0_0,?]");
  INFER_OK(op, "[2,1];?", "[d0_0,?]");
  Tensor num_samples = test::AsScalar<int32>(3);
  op.input_tensors[1] = &num_samples;
  INFER_OK(op, "[2,1];[]", "[d0_0,3]");
  num_samples = test::AsTensor<int32>({1, 2, 3});
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[2,1];[3]");
}

TEST(RandomOpsTest, RandomGamma_ShapeFn) {
  ShapeInferenceTestOp op("RandomGamma");
  op.input_tensors.resize(2);

  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "?;[3]", "?");
  INFER_OK(op, "[1];?", "?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2];[3,4]");
  Tensor shape = test::AsTensor<int32>({1, 2, 3});
  op.input_tensors[0] = &shape;
  INFER_OK(op, "[3];[4,?]", "[1,2,3,d1_0,d1_1]");
  INFER_OK(op, "[3];[4,5]", "[1,2,3,d1_0,d1_1]");
  INFER_OK(op, "[3];[]", "[1,2,3]");
}

}  // end namespace tensorflow
