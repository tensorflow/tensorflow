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

TEST(MathOpsTest, UnchangedShape_ShapeFn) {
  INFER_OK("Cast", "?", "in0");
  INFER_OK("Cast", "[?]", "in0");
  INFER_OK("Cast", "[1,?,3,4]", "in0");
}

TEST(MathOpsTest, FFT) {
  for (const auto* op : {"FFT", "IFFT"}) {
    INFER_OK(op, "?", "[?]");
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?]", "in0");
    INFER_OK(op, "[1]", "in0");
  }

  for (const auto* op : {"FFT2D", "IFFT2D"}) {
    INFER_OK(op, "?", "[?,?]");
    INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[1]");
    INFER_OK(op, "[?,1]", "in0");
    INFER_OK(op, "[1,2]", "in0");
  }

  for (const auto* op : {"FFT3D", "IFFT3D"}) {
    INFER_OK(op, "?", "[?,?,?]");
    INFER_ERROR("Shape must be rank 3 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?,1,?]", "in0");
    INFER_OK(op, "[1,2,3]", "in0");
  }

  for (const auto* op : {"BatchFFT", "BatchIFFT"}) {
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[]");
    INFER_OK(op, "[?]", "in0");
    INFER_OK(op, "[1]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }

  for (const auto* op : {"BatchFFT2D", "BatchIFFT2D"}) {
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
    INFER_OK(op, "[?,1]", "in0");
    INFER_OK(op, "[1,2]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }

  for (const auto* op : {"BatchFFT3D", "BatchIFFT3D"}) {
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 3 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?,1,?]", "in0");
    INFER_OK(op, "[1,2,3]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }
}

TEST(MathOpsTest, Segment) {
  // Tests SegmentReductionShapeFn.
  for (const auto* op : {"SegmentMax", "SegmentMean", "SegmentMin",
                         "SegmentProd", "SegmentSum"}) {
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "?;[100]", "?");

    // Data shape with single dimension.
    INFER_OK(op, "[?];?", "[?]");
    INFER_OK(op, "[?];[100]", "[?]");
    INFER_OK(op, "[1];?", "[?]");
    INFER_OK(op, "[1];[100]", "[?]");

    // Data shape with multiple dimensions.
    INFER_OK(op, "[?,?];?", "[?,d0_1]");
    INFER_OK(op, "[?,2];[100]", "[?,d0_1]");
    INFER_OK(op, "[?,2,?,4];[100]", "[?,d0_1,d0_2,d0_3]");
    INFER_OK(op, "[1,?];?", "[?,d0_1]");
    INFER_OK(op, "[1,2];[100]", "[?,d0_1]");
    INFER_OK(op, "[1,2,?,4];[100]", "[?,d0_1,d0_2,d0_3]");

    // Error cases.
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[1,2]");
    INFER_ERROR("Shape must have rank >= 1, but is 0", op, "[];[1]");
  }
}

TEST(MathOpsTest, BroadcastBinaryOps) {
  for (const auto* op : {"Add",        "Complex",
                         "Div",        "Equal",
                         "Greater",    "GreaterEqual",
                         "Igamma",     "Igammac",
                         "Zeta",       "Polygamma",
                         "Less",       "LessEqual",
                         "LogicalAnd", "LogicalOr",
                         "Maximum",    "Minimum",
                         "Mod",        "Mul",
                         "NotEqual",   "Pow",
                         "Sub",        "SquaredDifference"}) {
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[1,2];?", "?");
    INFER_OK(op, "?;[1,2]", "?");

    INFER_OK(op, "[?];[1]", "[d0_0]");
    INFER_OK(op, "[1];[?]", "[d1_0]");
    INFER_OK(op, "[?];[2]", "[d1_0]");
    INFER_OK(op, "[2];[?]", "[d0_0]");
    INFER_OK(op, "[?];[?]", "[?]");
    INFER_OK(op, "[];[?]", "[d1_0]");
    INFER_OK(op, "[?];[]", "[d0_0]");

    INFER_OK(op, "[1];[1]", "[d0_0|d1_0]");
    INFER_OK(op, "[];[1]", "[d1_0]");
    INFER_OK(op, "[1];[]", "[d0_0]");

    INFER_OK(op, "[2];[2]", "[d0_0|d1_0]");
    INFER_OK(op, "[];[2]", "[d1_0]");
    INFER_OK(op, "[1];[2]", "[d1_0]");
    INFER_OK(op, "[2];[1]", "[d0_0]");
    INFER_OK(op, "[2];[]", "[d0_0]");

    INFER_OK(op, "[0];[0]", "[d0_0|d1_0]");
    INFER_OK(op, "[];[0]", "[d1_0]");
    INFER_OK(op, "[1];[0]", "[d1_0]");
    INFER_OK(op, "[0];[1]", "[d0_0]");
    INFER_OK(op, "[0];[]", "[d0_0]");

    // Multiple dimension cases (same test cases, switching x and y).
    INFER_OK(op, "[?,1,2,3,4,5];[3,1,?]",
             "[d0_0,d0_1,d0_2,d0_3|d1_0,d0_4,d0_5]");
    INFER_OK(op, "[3,1,?];[?,1,2,3,4,5]",
             "[d1_0,d1_1,d1_2,d1_3|d0_0,d1_4,d1_5]");
  }
}

}  // end namespace tensorflow
