/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ReduceSliceOpsTest, ReduceSliceSum_ShapeFn) {
  ShapeInferenceTestOp op("ReduceSliceSum");
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "[10,20];[100,2];[]", "[?,?]");
  INFER_OK(op, "[10,20];[?,2];[]", "[?,?]");
  INFER_OK(op, "[10,20];[0];[]", "[?,?]");
  INFER_OK(op, "[10,20];[1];[]", "[?,?]");
  INFER_OK(op, "[10,20];[?];[]", "[?,?]");
  INFER_OK(op, "[?,?];[?,2];[]", "[?,?]");
  INFER_OK(op, "[?,?];[25,2];[]", "[?,?]");
  INFER_OK(op, "[?];[123,2];[]", "[?]");
  INFER_OK(op, "[1,2,3,4];[100,2];[]", "[?,?,?,?]");

  INFER_ERROR("must be rank 0", op, "?;[?,2];[?]");
  INFER_ERROR("must be at least rank 1", op, "?;[];[]");
  INFER_ERROR("must be at most rank 2", op, "?;[1,2,3];[]");
  INFER_ERROR("must be equal, but are 1 and 2", op, "?;[?,1];[]");
  INFER_ERROR("must be at least rank 1", op, "[];?;[]");
}

}  // end namespace tensorflow
