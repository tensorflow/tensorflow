/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
yo

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(MathOpsTest, AddN_ShapeFn) {
  INFER_OK("AddN", "?;?", "in0|in1");
  INFER_OK("AddN", "[1,?]", "in0");
  INFER_OK("AddN", "[1,2];[?,2]", "in0");
  INFER_OK("AddN", "[1,2];[1,2]", "in0|in1");
  INFER_OK("AddN", "[?,2];[1,2]", "in1");
  INFER_OK("AddN", "[1,?];[?,2];[1,2]", "in2");
  INFER_OK("AddN", "[1,2];[?,2];[1,?]", "in0");
  INFER_OK("AddN", "?;?;[1,2]", "in2");
  INFER_OK("AddN", "[1,?];[?,2]", "[d0_0,d1_1]");
  INFER_OK("AddN", "[?,2,?];[?,?,3]", "[d0_0|d1_0,d0_1,d1_2]");
  INFER_OK("AddN", "[?,2];[1,?]", "[d1_0,d0_1]");

  INFER_ERROR(("Dimension 1 in both shapes must be equal, but are 2 and "
               "4\n\tFrom merging shape 0 with other shapes."),
              "AddN", "[1,2];?;[1,4]");
  INFER_ERROR(("Shapes must be equal rank, but are 2 and 3\n\tFrom merging "
               "shape 1 with other shapes."),
              "AddN", "?;[1,2];?;[1,2,3]");
}

}  // end namespace tensorflow
