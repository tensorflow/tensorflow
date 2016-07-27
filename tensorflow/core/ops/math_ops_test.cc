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
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(MathOpsTest, AddN_ShapeFn) {
  ShapeInferenceTestOp op("AddN");
  auto set_n = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
    TF_CHECK_OK(NodeDefBuilder("test", "AddN")
                    .Input(src_list)
                    .Attr("N", n)
                    .Finalize(&op.node_def));
  };

  set_n(2);
  // Adding two unknowns returns either input.
  INFER_OK(op, "?;?", "in0|in1");

  // known+unknown returns the known input.
  INFER_OK(op, "[1];[?]", "in0");
  INFER_OK(op, "[1];?", "in0");
  INFER_OK(op, "[?];[1]", "in1");
  INFER_OK(op, "?;[1]", "in1");

  set_n(2);
  INFER_OK(op, "[1,2];[?,2]", "in0");
  INFER_OK(op, "[1,2];[1,2]", "in0|in1");
  INFER_OK(op, "[?,2];[1,2]", "in1");

  set_n(3);
  INFER_OK(op, "[1,?];[?,2];[1,2]", "in2");
  INFER_OK(op, "[1,2];[?,2];[1,?]", "in0");
  INFER_OK(op, "?;?;[1,2]", "in2");

  set_n(2);
  INFER_OK(op, "?;[1,2]", "in1");
  INFER_OK(op, "[1,?];[?,2]", "[d0_0,d1_1]");
  INFER_OK(op, "[?,2,?];[?,?,3]", "[d0_0|d1_0,d0_1,d1_2]");
  INFER_OK(op, "[?,2];[1,?]", "[d1_0,d0_1]");

  set_n(3);
  INFER_ERROR(("Dimension 1 in both shapes must be equal, but are 2 and "
               "4\n\tFrom merging shape 0 with other shapes."),
              op, "[1,2];?;[1,4]");
  set_n(4);
  INFER_ERROR(("Shapes must be equal rank, but are 2 and 3\n\tFrom merging "
               "shape 1 with other shapes."),
              op, "?;[1,2];?;[1,2,3]");
}

TEST(MathOpsTest, UnchangedShape_ShapeFn) {
  ShapeInferenceTestOp op("Cast");
  INFER_OK(op, "?", "in0");
  INFER_OK(op, "[?]", "in0");
  INFER_OK(op, "[1,?,3,4]", "in0");
}

TEST(MathOpsTest, FFT_ShapeFn) {
  for (const auto* op_name : {"FFT", "IFFT"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "[?]");
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?]", "in0");
    INFER_OK(op, "[1]", "in0");
  }

  for (const auto* op_name : {"FFT2D", "IFFT2D"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "[?,?]");
    INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[1]");
    INFER_OK(op, "[?,1]", "in0");
    INFER_OK(op, "[1,2]", "in0");
  }

  for (const auto* op_name : {"FFT3D", "IFFT3D"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "[?,?,?]");
    INFER_ERROR("Shape must be rank 3 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?,1,?]", "in0");
    INFER_OK(op, "[1,2,3]", "in0");
  }

  for (const auto* op_name : {"BatchFFT", "BatchIFFT"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[]");
    INFER_OK(op, "[?]", "in0");
    INFER_OK(op, "[1]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }

  for (const auto* op_name : {"BatchFFT2D", "BatchIFFT2D"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
    INFER_OK(op, "[?,1]", "in0");
    INFER_OK(op, "[1,2]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }

  for (const auto* op_name : {"BatchFFT3D", "BatchIFFT3D"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 3 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?,1,?]", "in0");
    INFER_OK(op, "[1,2,3]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }
}

TEST(MathOpsTest, Segment_ShapeFn) {
  // Tests SegmentReductionShapeFn.
  for (const auto* op_name : {"SegmentMax", "SegmentMean", "SegmentMin",
                              "SegmentProd", "SegmentSum"}) {
    ShapeInferenceTestOp op(op_name);
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
    INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[1]");
  }
}

TEST(MathOpsTest, BroadcastBinaryOps_ShapeFn) {
  for (const auto* op_name : {"Add",        "Complex",
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
    ShapeInferenceTestOp op(op_name);
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

TEST(MathOpsTest, Select_ShapeFn) {
  ShapeInferenceTestOp op("Select");
  INFER_OK(op, "?;?;?", "in1|in2");

  INFER_OK(op, "[];?;?", "in0");
  INFER_OK(op, "[1];?;?",
           "in1|in2");  // When cond is vector, t/e may not match it.
  INFER_OK(op, "[1,2];?;?", "in0");

  INFER_OK(op, "?;[];?", "in1");
  INFER_OK(op, "?;?;[]", "in2");
  INFER_OK(op, "?;[1];?", "in1");
  INFER_OK(op, "?;?;[1]", "in2");
  INFER_OK(op, "?;[1,2];?", "in1");
  INFER_OK(op, "?;?;[1,2]", "in2");

  INFER_ERROR("Shapes must be equal rank, but are 0 and 1", op, "[1];[];?");
  INFER_ERROR("Shapes must be equal rank, but are 1 and 0", op, "[];[1];?");
  INFER_ERROR("Shapes must be equal rank, but are 1 and 2", op, "[1,2];[1];?");
  INFER_OK(op, "[2];[?];[?]", "in0");

  INFER_OK(op, "[?];[?,?,3];[1,2,?]", "[d2_0,d2_1,d1_2]");
  INFER_OK(op, "[2];[?,?,3];[?,2,?]", "[d0_0,d2_1,d1_2]");
  INFER_ERROR("Dimensions must be equal, but are 2 and 1", op,
              "[1];[2,?,3];[?,2,?]");
  INFER_ERROR("Shapes must be equal rank, but are 3 and 2", op,
              "[2,?];[?,?,3];[?,2,?]");
  INFER_OK(op, "[2,?,?];[?,?,3];[?,2,?]", "[d0_0,d2_1,d1_2]");
  INFER_ERROR("Dimension 2 in both shapes must be equal, but are 3 and 5", op,
              "[2,?,5];[?,?,3];[?,2,?]");
}

TEST(MathOpsTest, Range_ShapeFn) {
  ShapeInferenceTestOp op("Range");
  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "[?]");
  INFER_ERROR("Shape must be rank 0 but is rank 2\n\t for 'start'", op,
              "[1,2];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 2\n\t for 'limit'", op,
              "?;[1,2];?");
  INFER_ERROR("Shape must be rank 0 but is rank 2\n\t for 'delta'", op,
              "?;?;[1,2]");

  Tensor start_t = test::AsScalar(1);
  op.input_tensors[0] = &start_t;
  INFER_OK(op, "?;?;?", "[?]");
  Tensor limit_t = test::AsScalar(1);
  op.input_tensors[1] = &limit_t;
  INFER_OK(op, "?;?;?", "[?]");

  Tensor delta_t = test::AsScalar(1);
  op.input_tensors[2] = &delta_t;
  INFER_OK(op, "?;?;?", "[0]");

  delta_t = test::AsScalar(0);
  INFER_ERROR("Requires delta > 0: 0", op, "?;?;?");
  delta_t = test::AsScalar(3);

  limit_t = test::AsScalar(-1);
  INFER_ERROR("Requires start <= limit: 1/-1", op, "?;?;?");

  limit_t = test::AsScalar(100);
  start_t = test::AsScalar(2);
  delta_t = test::AsScalar(3);
  INFER_OK(op, "?;?;?", "[33]");
}

TEST(MathOpsTest, LinSpace_ShapeFn) {
  ShapeInferenceTestOp op("LinSpace");
  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "[?]");
  INFER_ERROR("Shape must be rank 0 but is rank 2\n\t for 'start'", op,
              "[1,2];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 2\n\t for 'stop'", op,
              "?;[1,2];?");
  INFER_ERROR("Shape must be rank 0 but is rank 2\n\t for 'num'", op,
              "?;?;[1,2]");

  Tensor num_t = test::AsScalar(1);
  op.input_tensors[2] = &num_t;
  INFER_OK(op, "?;?;?", "[1]");
  num_t = test::AsScalar(2);
  INFER_OK(op, "?;?;?", "[2]");
  num_t = test::AsScalar(-1);
  INFER_ERROR("Requires num > 0: -1", op, "?;?;?");
}

TEST(MathOpsTest, UnsortedSegmentSum_ShapeFn) {
  ShapeInferenceTestOp op("UnsortedSegmentSum");
  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "?;[?];?", "?");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "?;?;[1,2]");
  INFER_ERROR("Dimensions must be equal, but are 2 and 3", op,
              "[1,?,2];[1,?,3];?");
  INFER_OK(op, "?;[3];?", "?");
  INFER_ERROR("Shape must be at least rank 3 but is rank 2", op,
              "[1,2];[1,2,3];?");

  Tensor num_segments_t = test::AsScalar(100);
  op.input_tensors[2] = &num_segments_t;
  INFER_OK(op, "[?,2,3,?,5];[1,2,?];[]", "[100,d0_3,d0_4]");

  num_segments_t = test::AsScalar(-1);
  INFER_ERROR(("Dimension size, given by scalar input 2, must be "
               "non-negative but is -1"),
              op, "[3];[3];?");
}

TEST(MathOpsTest, BatchMatMul_ShapeFn) {
  ShapeInferenceTestOp op("BatchMatMul");
  auto set_adj = [&op](bool adj_x, bool adj_y) {
    TF_CHECK_OK(NodeDefBuilder("test", "BatchMatMul")
                    .Input({"a", 0, DT_FLOAT})
                    .Input({"b", 0, DT_FLOAT})
                    .Attr("adj_x", adj_x)
                    .Attr("adj_y", adj_y)
                    .Finalize(&op.node_def));
  };

  set_adj(false, false);

  // Rank checks.
  INFER_ERROR("at least rank 3", op, "[1,2];?");
  INFER_ERROR("at least rank 3", op, "?;[1,2]");

  INFER_OK(op, "?;?", "?");

  // 2 batch dims.
  INFER_OK(op, "[?,?,?,?];?", "[d0_0,d0_1,d0_2,?]");

  // Test adj_a, testing output and that inner dims are compared.
  set_adj(false, false);
  INFER_OK(op, "[1,2,3,4];[1,2,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,3,1]");  // inner dim mismatch
  set_adj(true, false);
  INFER_OK(op, "[1,2,3,4];[1,2,?,?]", "[d0_0,d0_1,d0_3,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,3,1]");  // inner dim mismatch

  // Test adj_b=true.
  set_adj(false, true);
  INFER_OK(op, "[1,2,?,?];[1,2,3,4]", "[d0_0,d0_1,d0_2,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,1,3]");  // inner dim mismatch
  set_adj(true, true);
  INFER_OK(op, "[1,2,?,?];[1,2,3,4]", "[d0_0,d0_1,d0_3,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,1,3]");  // inner dim mismatch
}

}  // end namespace tensorflow
