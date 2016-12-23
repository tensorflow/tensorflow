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
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ArrayOpsTest, Pack_ShapeFn) {
  ShapeInferenceTestOp op("Pack");
  auto set_axis = [&op](int axis) {
    int n = 3;
    std::vector<NodeDefBuilder::NodeOut> src_list;
    for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input(src_list)
                     .Attr("N", n)
                     .Attr("axis", axis)
                     .Finalize(&op.node_def));
  };

  set_axis(0);
  INFER_OK(op, "?;?;?", "?");

  for (int axis : {0, -3}) {
    set_axis(axis);
    INFER_OK(op, "?;?;?", "?");
    INFER_OK(op, "[1,3];[1,3];?", "[3,d0_0|d1_0,d0_1|d1_1]");
    INFER_OK(op, "[?,3];[1,3];?", "[3,d1_0,d0_1|d1_1]");
    INFER_OK(op, "[?,?];[1,3];?", "[3,d1_0,d1_1]");
  }
  for (int axis : {1, -2}) {
    set_axis(axis);
    INFER_OK(op, "?;?;?", "?");
    INFER_OK(op, "[1,3];[1,3];?", "[d0_0|d1_0,3,d0_1|d1_1]");
    INFER_OK(op, "[?,3];[1,3];?", "[d1_0,3,d0_1|d1_1]");
    INFER_OK(op, "[?,?];[1,3];?", "[d1_0,3,d1_1]");
  }
  for (int axis : {2, -1}) {
    set_axis(axis);
    INFER_OK(op, "?;?;?", "?");
    INFER_OK(op, "[1,3];[1,3];?", "[d0_0|d1_0,d0_1|d1_1,3]");
    INFER_OK(op, "[?,3];[1,3];?", "[d1_0,d0_1|d1_1,3]");
    INFER_OK(op, "[?,?];[1,3];?", "[d1_0,d1_1,3]");
  }

  set_axis(-4);
  INFER_ERROR("Invalid axis: -4; must be in [-3,3)", op, "[1,3];[1,3];?");
  set_axis(3);
  INFER_ERROR("Invalid axis: 3; must be in [-3,3)", op, "[1,3];[1,3];?");

  set_axis(0);

  // Check that both components of error message are there.
  INFER_ERROR("Shapes must be equal rank, but are 3 and 2", op,
              "[1,2,3];?;[1,4]");
  INFER_ERROR("From merging shape 0 with other shapes.", op, "[1,2,3];?;[1,4]");
}

TEST(ArrayOpsTest, UnPack_ShapeFn) {
  ShapeInferenceTestOp op("Unpack");
  auto set_axis_and_num = [&op](int axis, int num) {
    TF_ASSERT_OK(NodeDefBuilder("test", "Unpack")
                     .Input("a", 0, DT_FLOAT)
                     .Attr("axis", axis)
                     .Attr("num", num)
                     .Finalize(&op.node_def));
  };

  set_axis_and_num(0, 1);
  INFER_OK(op, "?", "?");

  for (int axis : {0, -3}) {
    set_axis_and_num(axis, 1);
    INFER_OK(op, "?", "?");
    INFER_OK(op, "[1,2,3]", "[d0_1,d0_2]");
    INFER_OK(op, "[?,?,?]", "[d0_1,d0_2]");
  }
  for (int axis : {1, -2}) {
    set_axis_and_num(axis, 2);
    INFER_OK(op, "[1,2,3]", "[d0_0,d0_2];[d0_0,d0_2]");
    INFER_OK(op, "[?,?,?]", "[d0_0,d0_2];[d0_0,d0_2]");
  }
  for (int axis : {2, -1}) {
    set_axis_and_num(axis, 3);
    INFER_OK(op, "[1,2,3]", "[d0_0,d0_1];[d0_0,d0_1];[d0_0,d0_1]");
    INFER_OK(op, "[?,?,?]", "[d0_0,d0_1];[d0_0,d0_1];[d0_0,d0_1]");
  }

  set_axis_and_num(2, 2);
  INFER_ERROR("Dimension must be 2 but is 3", op, "[1,2,3]");

  set_axis_and_num(-4, 3);
  INFER_ERROR("Invalid axis: -4; must be in [-3,3)", op, "[1,2,3]");
  set_axis_and_num(3, 3);
  INFER_ERROR("Invalid axis: 3; must be in [-3,3)", op, "[1,2,3]");
}

TEST(ArrayOpsTest, Const_ShapeFn) {
  ShapeInferenceTestOp op("Const");
  TensorProto tensor_proto;
  auto* shape_proto = tensor_proto.mutable_tensor_shape();
  auto rebuild_node_def = [&op, &tensor_proto]() {
    TF_ASSERT_OK(NodeDefBuilder("test", "Const")
                     .Attr("value", tensor_proto)
                     .Finalize(&op.node_def));
  };

  TensorShape{}.AsProto(shape_proto);
  rebuild_node_def();
  INFER_OK(op, "", "[]");
  TensorShape{1, 2, 3, 4}.AsProto(shape_proto);
  rebuild_node_def();
  INFER_OK(op, "", "[1,2,3,4]");

  shape_proto->add_dim()->set_size(-1);
  rebuild_node_def();
  INFER_ERROR("Shape [1,2,3,4,-1] has negative dimensions", op, "");
}

TEST(ArrayOpsTest, UnchangedShapes_ShapeFn) {
  for (const char* op_name : {
           "CheckNumerics", "Identity", "QuantizeAndDequantize", "RefIdentity",
           "StopGradient", "ZerosLike",
       }) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "in0");
    INFER_OK(op, "[]", "in0");
    INFER_OK(op, "[1,2,?,4,5]", "in0");
  }

  // inputs 1 and 2 are ignored; input 0 is transferred to output 0.
  ShapeInferenceTestOp op("MatrixBandPart");
  INFER_OK(op, "?;?;?", "in0");
  INFER_OK(op, "[];?;?", "in0");
  INFER_OK(op, "[1,2,?,4,5];?;?", "in0");
}

TEST(ArrayOpsTest, Identity_ShapeFnHandles) {
  const char* op_name = "Identity";
  ShapeInferenceTestOp op(op_name);
  // Check that handle dtypes are preserved.
  const OpRegistrationData* op_reg_data;
  TF_ASSERT_OK(OpRegistry::Global()->LookUp(op.name, &op_reg_data));
  shape_inference::InferenceContext c(&op.node_def, op_reg_data->op_def,
                                      {TensorShapeProto()}, {}, {}, {},
                                      {DT_BOOL});
  TF_ASSERT_OK(c.construction_status());
  ASSERT_TRUE(op_reg_data->shape_inference_fn != nullptr);
  TF_ASSERT_OK(c.Run(op_reg_data->shape_inference_fn));
  EXPECT_TRUE(c.output_handle_dtype(0) == DT_BOOL);
}

TEST(ArrayOpsTest, Diag_ShapeFn) {
  ShapeInferenceTestOp op("Diag");
  INFER_OK(op, "?", "?");
  INFER_OK(op, "[]", "[]");
  INFER_OK(op, "[1,?,3]", "[d0_0,d0_1,d0_2,d0_0,d0_1,d0_2]");
  INFER_ERROR("Shape must be at most rank 3 but is rank 4", op, "[?,1,2,3]");
}

TEST(ArrayOpsTest, DiagPart_ShapeFn) {
  ShapeInferenceTestOp op("DiagPart");
  INFER_OK(op, "?", "?");
  INFER_OK(op, "[]", "[]");
  INFER_OK(op, "[1,?,?,4]", "[d0_0,d0_3]");
  INFER_OK(op, "[1,?,3,?,4,3]", "[d0_0,d0_4,d0_2|d0_5]");
  INFER_ERROR("Input must have even rank <= 6, input rank is 1", op, "[?]");
  INFER_ERROR("Input must have even rank <= 6, input rank is 3", op, "[1,2,3]");
  INFER_ERROR("Input must have even rank <= 6, input rank is 8", op,
              "[1,2,3,?,?,?,?,?]");
  INFER_ERROR("Dimensions must be equal, but are 2 and 10", op, "[1,2,?,10]");
}

TEST(ArrayOpsTest, MatrixDiag_ShapeFn) {
  ShapeInferenceTestOp op("MatrixDiag");
  INFER_OK(op, "?", "?");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[]");
  INFER_OK(op, "[?]", "[d0_0,d0_0]");
  INFER_OK(op, "[1,?,?,4]", "[d0_0,d0_1,d0_2,d0_3,d0_3]");
}

TEST(ArrayOpsTest, MatrixDiagPart_ShapeFn) {
  ShapeInferenceTestOp op("MatrixDiagPart");
  INFER_OK(op, "?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[?]");
  INFER_OK(op, "[?,1,2,2]", "[d0_0,d0_1,d0_2|d0_3]");
  INFER_OK(op, "[?,1,2,3]", "[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,1,3,2]", "[d0_0,d0_1,d0_3]");
}

TEST(ArrayOpsTest, Reverse_ShapeFn) {
  ShapeInferenceTestOp op("Reverse");
  INFER_OK(op, "?;?", "in0");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "?;[]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[?,2]");
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];[4]");
  INFER_ERROR("reverse does not work on tensors with more than 8 dimensions",
              op, "[1,2,3,4,5,6,7,8,9];[9]");
  INFER_OK(op, "[1,2,3,?];[4]", "in0");
  INFER_OK(op, "[1,2,3,?,5,6,7,8];[8]", "in0");
}

TEST(ArrayOpsTest, ReverseV2_ShapeFn) {
  ShapeInferenceTestOp op("ReverseV2");
  INFER_OK(op, "?;?", "in0");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "?;[]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[?,2]");
  INFER_OK(op, "[1,2,3];[2]", "in0");
  INFER_ERROR("reverse does not work on tensors with more than 8 dimensions",
              op, "[1,2,3,4,5,6,7,8,9];[9]");
  INFER_OK(op, "[1,2,3,?];[4]", "in0");
  INFER_OK(op, "[1,2,3,?,5,6,7,8];[8]", "in0");
}

TEST(ArrayOpsTest, Fill_ShapeFn) {
  ShapeInferenceTestOp op("Fill");
  op.input_tensors.resize(2);
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[?];?", "?");
  INFER_OK(op, "[4];?", "[?,?,?,?]");

  Tensor in_t = test::AsTensor<int32>({1, 2, 3, 4});
  op.input_tensors[0] = &in_t;
  INFER_OK(op, "[4];?", "[1,2,3,4]");
}

TEST(ArrayOpsTest, Gather_ShapeFn) {
  ShapeInferenceTestOp op("Gather");
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[1,?,2];[3]", "[d1_0,d0_1,d0_2]");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[1,2,3]");
}

TEST(ArrayOpsTest, GatherNd_ShapeFn) {
  ShapeInferenceTestOp op("GatherNd");

  // Inputs are (params, indices).
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[1,?,3,?];[?,0]", "[d1_0,d0_0,d0_1,d0_2,d0_3]");
  INFER_OK(op, "[1,?,3,?];[?,4]", "[d1_0]");

  // params.rank >= indices.dim(-1).
  INFER_ERROR("indices.shape[-1] must be <= params.rank", op, "[1,2,3];[4]");
}

TEST(ArrayOpsTest, Shape_ShapeFn) {
  ShapeInferenceTestOp op("Shape");
  INFER_OK(op, "?", "[?]");
  INFER_OK(op, "[?]", "[1]");
  INFER_OK(op, "[?,2,3,4,5]", "[5]");
}

TEST(ArrayOpsTest, ShapeN_ShapeFn) {
  ShapeInferenceTestOp op("ShapeN");
  int n = 3;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
  TF_ASSERT_OK(NodeDefBuilder("test", "ShapeN")
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));
  INFER_OK(op, "?;?;?", "[?];[?];[?]");
  INFER_OK(op, "[?];[?];[?]", "[1];[1];[1]");
  INFER_OK(op, "[?,2,3,4,5];?;[1,?,3]", "[5];[?];[3]");
}

TEST(ArrayOpsTest, Unique_ShapeFn) {
  ShapeInferenceTestOp op("Unique");
  INFER_OK(op, "?", "[?];in0");
  INFER_OK(op, "[1,2,3,?,5]", "[?];in0");
}

TEST(ArrayOpsTest, UniqueWithCounts_ShapeFn) {
  ShapeInferenceTestOp op("UniqueWithCounts");
  INFER_OK(op, "?", "[?];in0;[?]");
  INFER_OK(op, "[1,2,3,?,5]", "[?];in0;[?]");
}

TEST(ArrayOpsTest, InvertPermutation_ShapeFn) {
  ShapeInferenceTestOp op("InvertPermutation");
  INFER_OK(op, "?", "[?]");
  INFER_OK(op, "[1]", "in0");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[]");
}

TEST(ArrayOpsTest, PadD_ShapeFn) {
  for (const char* op_name : {"Pad", "MirrorPad"}) {
    ShapeInferenceTestOp op(op_name);
    op.input_tensors.resize(2);

    // Inputs are input and paddings.

    INFER_OK(op, "?;?", "?");

    // Check shape of paddings.
    INFER_ERROR("Shape must be rank 2 but is rank 3", op, "?;[1,2,3]");
    INFER_ERROR("Dimension must be 2 but is 4", op, "?;[1,4]");

    // input.rank and paddings.dim(0) are equal. This is the number of dims in
    // output.
    INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];[4,2]");
    INFER_OK(op, "[1,2,3];?", "[?,?,?]");
    INFER_OK(op, "?;[3,2]", "[?,?,?]");

    // Make the paddings tensor known and verify padding values get added.
    // E.g., if padding is ((1,10),(2,20),(3,30)) then values 11,22,23 are added
    // to input dims to get output.
    Tensor paddings_t(DT_INT64, TensorShape{3, 2});
    test::FillValues<int64>(&paddings_t, {1, 10, 2, 20, 3, 30});
    op.input_tensors[1] = &paddings_t;
    INFER_OK(op, "[100,200,300];[3,2]", "[111,222,333]");
    INFER_OK(op, "[100,?,300];[3,2]", "[111,?,333]");
    INFER_OK(op, "?;[3,2]", "[?,?,?]");
  }
}

TEST(ArrayOpsTest, MirrorPadGrad_ShapeFn) {
  ShapeInferenceTestOp op("MirrorPadGrad");
  op.input_tensors.resize(2);

  // Inputs are input and paddings.
  INFER_OK(op, "?;?", "?");

  // First padding dimension is unknown, so rank is unknown.
  INFER_OK(op, "?;[?,4]", "?");

  // Input tensor rank doesn't match paddings dimension.
  INFER_ERROR("must be rank 3 but is rank 2", op, "[?,?];[3,2]");

  // Paddings tensor is not a [rank x 2] matrix.
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 3 and 2", op,
              "[?,?,?];[3,3]");

  // Paddings tensor is unknown, but rank is known, so the output
  // shape is a rank 3 unknown shape.
  INFER_OK(op, "[?,?,?];[3,2]", "[?,?,?]");

  // Make the paddings tensor known and verify padding values get
  // subtracted.  E.g., if padding is ((1,10),(2,20),(3,30)) then
  // values 11,22,23 are subtracted to input dims to get output.
  Tensor paddings_t(DT_INT64, TensorShape{3, 2});
  test::FillValues<int64>(&paddings_t, {1, 10, 2, 20, 3, 30});
  op.input_tensors[1] = &paddings_t;

  INFER_OK(op, "[111,222,333];[3,2]", "[100,200,300]");
  INFER_OK(op, "[111,?,333];[3,2]", "[100,?,300]");
}

TEST(ArrayOpsTest, BroadcastArgs_ShapeFn) {
  ShapeInferenceTestOp op("BroadcastArgs");
  INFER_OK(op, "?;?", "[?]");
  INFER_OK(op, "[123];[1]", "[123]");
  INFER_OK(op, "[1];[123]", "[123]");
  INFER_OK(op, "[123];[121]", "[123]");

  // Rank checks
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "?;[]");
}

TEST(ArrayOpsTest, BroadcastGradientArgs_ShapeFn) {
  ShapeInferenceTestOp op("BroadcastGradientArgs");
  // Output is always two unknown vectors.
  INFER_OK(op, "?;?", "[?];[?]");
  INFER_OK(op, "[123];[456]", "[?];[?]");

  // Rank checks
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "?;[]");
}

TEST(ArrayOpsTest, ListDiff_ShapeFn) {
  ShapeInferenceTestOp op("BroadcastGradientArgs");
  // Output is always two matching unknown vectors.
  INFER_OK(op, "?;?", "[?];[?]");
  INFER_OK(op, "[123];[456]", "[?];[?]");

  // Rank checks
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "?;[]");
}

TEST(ArrayOpsTest, MatrixSetDiag_ShapeFn) {
  ShapeInferenceTestOp op("MatrixSetDiag");

  // Inputs are input and diagonal.

  // Rank checks.
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1];?");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "?;[]");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[2,2];[]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[2,2];[2,2]");

  // diagonal[-1] must match smallest matrix dimension.
  INFER_ERROR("Dimensions must be equal, but are 2 and 3", op, "[2,3];[3]");

  // Output matches input.
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[1,2,2];[1,2]", "in0");
  INFER_OK(op, "[1,2,3];?", "in0");
  INFER_OK(op, "[1,3,2];?", "in0");
  INFER_OK(op, "[1,?,2];[?,?]", "in0");
  INFER_OK(op, "[1,?,?];[?,2]", "in0");

  // Infer batch shape from diag when input is not fully specified.
  INFER_OK(op, "?;[1,2]", "[d1_0,?,?]");
  INFER_OK(op, "[?,?,3];[1,2]", "[d1_0,d0_1,d0_2]");
  INFER_OK(op, "[?,3,?];[1,2]", "[d1_0,d0_1,d0_2]");
  INFER_OK(op, "[?,3,2];[1,2]", "[d1_0,d0_1,d0_2]");
}

TEST(ArrayOpsTest, ExpandDims_ShapeFn) {
  ShapeInferenceTestOp op("ExpandDims");
  op.input_tensors.resize(2);

  // With unknown dim tensor value, output is unknown.
  INFER_OK(op, "?;?", "?");
  Tensor dim_t;
  op.input_tensors[1] = &dim_t;

  // Expand at front of tensor.
  for (int32 idx : {0, -4}) {
    dim_t = test::AsScalar<int32>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[1,d0_0,d0_1,d0_2]");
  }

  // Expand at middle of tensor.
  for (int32 idx : {1, -3}) {
    dim_t = test::AsScalar<int32>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[d0_0,1,d0_1,d0_2]");

    // Repeat with int64.
    dim_t = test::AsScalar<int64>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[d0_0,1,d0_1,d0_2]");
  }
  for (int32 idx : {2, -2}) {
    dim_t = test::AsScalar<int32>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[d0_0,d0_1,1,d0_2]");

    // Repeat with int64.
    dim_t = test::AsScalar<int64>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[d0_0,d0_1,1,d0_2]");
  }

  for (int32 idx : {3, -1}) {
    // Expand at the end.
    dim_t = test::AsScalar<int32>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[d0_0,d0_1,d0_2,1]");

    // Repeat with int64.
    dim_t = test::AsScalar<int64>(idx);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[5,?,7];?", "[d0_0,d0_1,d0_2,1]");
  }
  for (int32 idx : {4, -5}) {
    // Invalid idx.
    dim_t = test::AsScalar<int32>(idx);
    INFER_ERROR("not in the interval [-4, 3]", op, "[5,?,7];?");
    dim_t = test::AsScalar<int64>(idx);
    INFER_ERROR("not in the interval [-4, 3]", op, "[5,?,7];?");
  }

  // Expand using an input vector tensor.
  std::vector<int32> dims;
  dims.push_back(0);
  dim_t = test::AsTensor<int32>(dims);
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[5,?,7];?", "[1,d0_0,d0_1,d0_2]");

  // Expand using too many input elements.
  dims.push_back(1);
  dim_t = test::AsTensor<int32>(dims);
  INFER_ERROR("'dim' input must be a tensor with a single", op, "?;?");
  INFER_ERROR("'dim' input must be a tensor with a single", op, "[5,6,7];?");

  // Examples from ExpandDims doc.
  dim_t = test::AsScalar<int32>(0);
  INFER_OK(op, "[2];[]", "[1,d0_0]");
  dim_t = test::AsScalar<int32>(1);
  INFER_OK(op, "[2];[]", "[d0_0,1]");
  dim_t = test::AsScalar<int32>(-1);
  INFER_OK(op, "[2];[]", "[d0_0,1]");
}

TEST(ArrayOpsTest, ImmutableConst_ShapeFn) {
  ShapeInferenceTestOp op("ImmutableConst");

  TF_ASSERT_OK(NodeDefBuilder("test", "ImmutableConst")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", TensorShape({1, 2, 3}))
                   .Attr("memory_region_name", "test_region")
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "[1,2,3]");

  TF_ASSERT_OK(NodeDefBuilder("test", "ImmutableConst")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", TensorShape({}))
                   .Attr("memory_region_name", "test_region")
                   .Finalize(&op.node_def));
  INFER_OK(op, "", "[]");

  TF_ASSERT_OK(NodeDefBuilder("test", "ImmutableConst")
                   .Attr("dtype", DT_FLOAT)
                   .Attr("shape", "invalid")
                   .Attr("memory_region_name", "test_region")
                   .Finalize(&op.node_def));
  INFER_ERROR("AttrValue had value with type 'string' when 'shape' expected",
              op, "");
}

TEST(ArrayOpsTest, Concat_ShapeFn) {
  ShapeInferenceTestOp op("Concat");
  auto set_n = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
    TF_ASSERT_OK(NodeDefBuilder("test", "Concat")
                     .Input({"concat_dim", 0, DT_INT32})
                     .Input(src_list)
                     .Attr("n", n)
                     .Finalize(&op.node_def));
  };

  // Confirm dimension[0] of the input (the concat_dim) is a scalar.
  set_n(2);
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[1];?;?");

  // Test with the input concat_dim tensor not known. This takes the known rank
  // of the inputs and makes a tensor of that many unknown dims.
  set_n(7);
  INFER_OK(op, "?;?;?;?;[1,2,3];?;[3,2,1];?", "[?,?,?]");
  set_n(4);
  INFER_OK(op, "?;?;?;[1,2,3,4];[4,3,2,1]", "[?,?,?,?]");
  INFER_OK(op, "?;?;?;?;?", "?");  // output rank unknown
  INFER_ERROR("Can't concatenate scalars (use tf.pack instead)", op,
              "?;?;?;[];[]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "?;?;?;[1,2];[1,2,3]");

  // Test when the concat_dim tensor is known. The concatenated dimension is
  // summed across all input tensors, and other dimensions are merged.
  Tensor concat_dim_t;
  op.input_tensors.push_back(&concat_dim_t);
  set_n(2);

  // Sum dim 0, merge the other two dims.
  for (int concat_dim : {0, -3}) {
    concat_dim_t = test::AsScalar(concat_dim);
    INFER_OK(op, "[];[100,2,?];[10,?,3]", "[110,d1_1,d2_2]");
    INFER_ERROR("Dimension 1 in both shapes must be equal, but are 5 and 3", op,
                "[];[100,2,5];[10,?,3]");
    // concat_dim can't be summed, as one value is unknown.
    INFER_OK(op, "[];[100,2,?];[?,?,3]", "[?,d1_1,d2_2]");
    INFER_OK(op, "[];[?,2,?];[10,?,3]", "[?,d1_1,d2_2]");
  }

  // Test with a higher concat_dim.
  for (bool use_negative : {false, true}) {
    concat_dim_t = test::AsScalar(use_negative ? -2 : 1);
    INFER_OK(op, "[];[1,100,?];[?,10,3]", "[d1_0,110,d2_2]");
    concat_dim_t = test::AsScalar(use_negative ? -1 : 1);
    INFER_OK(op, "[];[1,100];[?,10]", "[d1_0,110]");
    INFER_OK(op, "[];[?,100];[1,10]", "[d2_0,110]");

    // concat_dim is out of bounds.
    concat_dim_t = test::AsScalar(use_negative ? -2 : 1);
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
                "[];[100];[10,?]");
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
                "[];[100,5];[10]");
  }

  // concat_dim is too low.
  concat_dim_t = test::AsScalar(-2);
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[];[100];[10,?]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[];[100,5];[10]");

  // Repeat successful case with several unknown inputs.
  set_n(5);
  concat_dim_t = test::AsScalar(1);
  INFER_OK(op, "[];?;[1,100,?];[?,?,?];[?,10,3];?", "[d2_0,?,d4_2]");
}

TEST(ArrayOpsTest, ConcatV2_ShapeFn) {
  ShapeInferenceTestOp op("ConcatV2");
  auto set_n = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
    TF_ASSERT_OK(NodeDefBuilder("test", "ConcatV2")
                     .Input(src_list)
                     .Input({"axis", 0, DT_INT32})
                     .Attr("n", n)
                     .Finalize(&op.node_def));
  };

  // Confirm dimension[0] of the input (the concat_dim) is a scalar.
  set_n(2);
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[1]");

  // Test with the input concat_dim tensor not known. This takes the known rank
  // of the inputs and makes a tensor of that many unknown dims.
  set_n(7);
  INFER_OK(op, "?;?;?;?;[1,2,3];?;[3,2,1];?", "[?,?,?]");
  set_n(4);
  INFER_OK(op, "?;?;[1,2,3,4];[4,3,2,1];?", "[?,?,?,?]");
  INFER_OK(op, "?;?;?;?;?", "?");  // output rank unknown
  INFER_ERROR("Can't concatenate scalars (use tf.pack instead)", op,
              "?;?;[];[];?");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "?;?;[1,2];[1,2,3];?");

  // Test when the concat_dim tensor is known. The concatenated dimension is
  // summed across all input tensors, and other dimensions are merged.
  Tensor concat_dim_t;
  op.input_tensors.resize(3);
  op.input_tensors[2] = &concat_dim_t;

  set_n(2);

  // Invalid concat dim value.
  // concat_dim_t = test::AsScalar(-1);
  // INFER_ERROR("Expected concat_dim >= 0, but got -1", op, "?;?;?");

  // Sum dim 0, merge the other two dims.
  concat_dim_t = test::AsScalar(0);
  INFER_OK(op, "[100,2,?];[10,?,3];[]", "[110,d0_1,d1_2]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 5 and 3", op,
              "[100,2,5];[10,?,3];[]");
  // concat_dim can't be summed, as one value is unknown.
  INFER_OK(op, "[100,2,?];[?,?,3];[]", "[?,d0_1,d1_2]");
  INFER_OK(op, "[?,2,?];[10,?,3];[]", "[?,d0_1,d1_2]");

  // Test with a higher concat_dim.
  concat_dim_t = test::AsScalar(1);
  INFER_OK(op, "[1,100,?];[?,10,3];[]", "[d0_0,110,d1_2]");
  INFER_OK(op, "[1,100];[?,10];[]", "[d0_0,110]");
  INFER_OK(op, "[?,100];[1,10];[]", "[d1_0,110]");
  // concat_dim is too high.
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[100];[10,?];[]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[100,5];[10];[]");
  // concat_dim is too low.
  concat_dim_t = test::AsScalar(-2);
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[100];[10,?];[]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[100,5];[10];[]");

  // Repeat successful case with several unknown inputs.
  op.input_tensors.resize(6);
  op.input_tensors[3] = nullptr;
  op.input_tensors[5] = &concat_dim_t;
  concat_dim_t = test::AsScalar(1);

  set_n(5);
  INFER_OK(op, "?;[1,100,?];[?,?,?];[?,10,3];?;[]", "[d1_0,?,d3_2]");
}

TEST(ArrayOpsTest, ConcatOffset_ShapeFn) {
  ShapeInferenceTestOp op("ConcatOffset");

  const int n = 4;
  std::vector<NodeDefBuilder::NodeOut> src_list;
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_INT32);
  TF_ASSERT_OK(NodeDefBuilder("test", "ConcatOffset")
                   .Input({"concat_dim", 0, DT_INT32})
                   .Input(src_list)
                   .Attr("n", n)
                   .Finalize(&op.node_def));
  INFER_OK(op, "?;?;?;?;?", "in1;in2;in3;in4");
}

TEST(ArrayOpsTest, Reshape_ShapeFn) {
  ShapeInferenceTestOp op("Reshape");
  op.input_tensors.resize(2);

  // No valid shape provided.
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[?];?", "?");
  INFER_OK(op, "[?];[?]", "?");
  INFER_OK(op, "[4];[?]", "?");

  // All dimensions provided.
  Tensor new_shape = test::AsTensor<int32>({1, 2, 3});
  op.input_tensors[1] = &new_shape;
  INFER_OK(op, "[?];[3]", "[1,2,3]");
  INFER_OK(op, "[6];[3]", "[1,2,3]");
  // The number of elements should match for the reshape to succeed.
  INFER_ERROR(
      "Cannot reshape a tensor with 12 elements to shape [1,2,3] (6 elements)",
      op, "[3,4];[3]");

  // Unknown dimensions.
  // Flatten:
  new_shape = test::AsTensor<int32>({-1});
  INFER_OK(op, "[?];[1]", "[?]");
  INFER_OK(op, "[2,2];[1]", "[4]");
  // The first dimension is inferred:
  new_shape = test::AsTensor<int32>({2, -1});
  INFER_OK(op, "[3,4];[2]", "[2,6]");
  // The total number of elements must be evenly divisible by the known
  // dimensions.
  INFER_ERROR("Dimension size must be evenly divisible by 2 but is 7", op,
              "[7];[2]");
  // Multiple missing dimensions cannot be inferred.
  new_shape = test::AsTensor<int32>({-1, -1, 2});
  INFER_OK(op, "[8];[3]", "[?,?,2]");

  // Reshaping to a scalar.
  new_shape = test::AsTensor<int32>({});
  INFER_OK(op, "[1];[0]", "[]");
  INFER_ERROR(
      "Cannot reshape a tensor with 2 elements to shape [] (1 elements)", op,
      "[1,2];[0]");

  // Reshaping a tensor with no elements.
  new_shape = test::AsTensor<int32>({-1});
  INFER_OK(op, "[0];[1]", "[0]");
  new_shape = test::AsTensor<int32>({-1, 6});
  INFER_OK(op, "[0,2];[1]", "[0,6]");
  new_shape = test::AsTensor<int32>({0, -1});
  INFER_OK(op, "[0,2];[1]", "[0,?]");
}

TEST(ArrayOpsTest, QuantizedReshape_ShapeFn) {
  ShapeInferenceTestOp op("QuantizedReshape");
  op.input_tensors.resize(2);

  // First test a subset of the Reshape_ShapeFn tests. Not all are tested, as
  // QuantizedReshape uses the same code for the reshape part of the operation.
  INFER_OK(op, "?;?;?;?", "?;[];[]");
  INFER_OK(op, "[?];?;?;?", "?;[];[]");
  INFER_OK(op, "[?];[?];?;?", "?;[];[]");
  INFER_OK(op, "[4];[?];?;?", "?;[];[]");
  Tensor new_shape = test::AsTensor<int32>({1, 2, 3});
  op.input_tensors[1] = &new_shape;
  INFER_OK(op, "[?];[3];?;?", "[1,2,3];[];[]");
  INFER_OK(op, "[6];[3];?;?", "[1,2,3];[];[]");
  INFER_ERROR(
      "Cannot reshape a tensor with 12 elements to shape [1,2,3] (6 elements)",
      op, "[3,4];[3];?;?");

  // Test the scalar rank checks on input_min and input_max.
  INFER_ERROR("must be rank 0", op, "?;?;[1];?");
  INFER_ERROR("must be rank 0", op, "?;?;?;[1]");
}

TEST(ArrayOpsTest, Placeholder_ShapeFn) {
  {
    // 2D shape
    ShapeInferenceTestOp op("Placeholder");
    TensorShape shape({1, 2});
    TF_ASSERT_OK(NodeDefBuilder("test", "Placeholder")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "[1,2]");
  }

  {
    // Scalar shapes are unknown shapes due to legacy.
    ShapeInferenceTestOp op("Placeholder");
    TensorShape shape({});
    TF_ASSERT_OK(NodeDefBuilder("test", "Placeholder")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "?");
  }

  {
    // Partial shape
    ShapeInferenceTestOp op("Placeholder");
    const int64 dims[2] = {1, -1};
    PartialTensorShape shape;
    TF_ASSERT_OK(PartialTensorShape::MakePartialShape(dims, 2, &shape));
    TF_ASSERT_OK(NodeDefBuilder("test", "Placeholder")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "[1,?]");
  }

  {
    ShapeInferenceTestOp op("PlaceholderWithDefault");
    const int64 dims[2] = {1, -1};
    PartialTensorShape shape;
    TF_ASSERT_OK(PartialTensorShape::MakePartialShape(dims, 2, &shape));
    TF_ASSERT_OK(NodeDefBuilder("test", "PlaceholderWithDefault")
                     .Input("input", 0, DT_FLOAT)
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "[1,2]", "[1,?]");

    // input shape is not compatible with output shape.
    INFER_ERROR("Dimension 0 in both shapes must be equal, but are 2 and 1", op,
                "[2,3]");
    // Wrong rank
    INFER_ERROR("Shapes must be equal rank, but are 3 and 2", op, "[1,3,10]");
  }
}

TEST(ArrayOpsTest, PlaceholderV2_ShapeFn) {
  {
    // 2D shape
    ShapeInferenceTestOp op("PlaceholderV2");
    TensorShape shape({1, 2});
    TF_ASSERT_OK(NodeDefBuilder("test", "PlaceholderV2")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "[1,2]");
  }

  {
    // Scalar shapes are supported in V2.
    ShapeInferenceTestOp op("PlaceholderV2");
    TensorShape shape({});
    TF_ASSERT_OK(NodeDefBuilder("test", "PlaceholderV2")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "[]");
  }

  {
    // Partial shape
    ShapeInferenceTestOp op("PlaceholderV2");
    const int64 dims[2] = {1, -1};
    PartialTensorShape shape;
    TF_ASSERT_OK(PartialTensorShape::MakePartialShape(dims, 2, &shape));
    TF_ASSERT_OK(NodeDefBuilder("test", "PlaceholderV2")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "[1,?]");
  }

  {
    // Unknown shape
    ShapeInferenceTestOp op("PlaceholderV2");
    PartialTensorShape shape;
    TF_ASSERT_OK(NodeDefBuilder("test", "PlaceholderV2")
                     .Attr("shape", shape)
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(&op.node_def));
    INFER_OK(op, "", "?");
  }
}

TEST(ArrayOpsTest, Transpose_ShapeFn) {
  ShapeInferenceTestOp op("Transpose");
  op.input_tensors.resize(2);

  // Missing shape information.
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "?;[?]", "?");
  INFER_OK(op, "?;[2]", "[?,?]");
  INFER_OK(op, "[?];?", "[?]");
  INFER_OK(op, "[?,?];[2]", "[?,?]");
  INFER_ERROR("Dimension must be 3 but is 2", op, "[1,2,3];[2]");
  Tensor perm = test::AsTensor<int32>({0});
  op.input_tensors[1] = &perm;
  INFER_OK(op, "[?];[?]", "[d0_0]");
  perm = test::AsTensor<int32>({1, 0});
  INFER_OK(op, "?;[2]", "[?,?]");
  INFER_OK(op, "[?,?];[2]", "[d0_1,d0_0]");
  INFER_OK(op, "[1,?];[2]", "[d0_1,d0_0]");

  // Invalid arguments.
  perm = test::AsTensor<int32>({1, 2});
  INFER_ERROR("perm dim 2 is out of range of input rank 2", op, "[1,2];[2]");
  perm = test::AsTensor<int32>({0});
  INFER_ERROR("Dimension must be 2 but is 1", op, "[1,2];[1]");

  // Larger valid cases.
  perm = test::AsTensor<int32>({1, 0, 3, 4, 2});
  INFER_OK(op, "[0,1,2,3,4];[5]", "[d0_1,d0_0,d0_3,d0_4,d0_2]");
  INFER_OK(op, "[0,?,2,3,4];[5]", "[d0_1,d0_0,d0_3,d0_4,d0_2]");
}

TEST(ArrayOpsTest, Bitcast_ShapeFn) {
  ShapeInferenceTestOp op("Bitcast");
  auto rebuild_node_def = [&op](DataType input_type, DataType output_type) {
    TF_ASSERT_OK(NodeDefBuilder("test", "Bitcast")
                     .Input("input", 0, input_type)
                     .Attr("type", output_type)
                     .Finalize(&op.node_def));
  };

  rebuild_node_def(DT_FLOAT, DT_INT32);
  // No valid shape provided, so output is unknown.
  INFER_OK(op, "?", "?");

  // Bitcasting from two equal sizes propagates shape.
  INFER_OK(op, "[1,2]", "in0");

  // Bitcasting from smaller to larger reduces the size of the last dimension.
  rebuild_node_def(DT_INT32, DT_INT64);
  INFER_OK(op, "[1,2]", "[d0_0]");  // last dimension matches divisor.
  // TODO(vrv): Seems like a bug, or at least, too lenient.
  INFER_OK(op, "[1,?]", "[d0_0]");
  // 4 is divisible by 2, but the shape function signature requires
  // that the last dimension matches the last value exactly.
  INFER_ERROR("does not match", op, "[1,4]");
  INFER_ERROR("does not match", op, "[1,3]");

  // Bitcasting from a larger type to a smaller type extends the dimension
  rebuild_node_def(DT_INT64, DT_INT32);
  INFER_OK(op, "[4,5]", "[d0_0,d0_1,2]");
  rebuild_node_def(DT_COMPLEX128, DT_INT32);
  INFER_OK(op, "[4,5]", "[d0_0,d0_1,4]");
  rebuild_node_def(DT_COMPLEX128, DT_HALF);
  INFER_OK(op, "[4,5]", "[d0_0,d0_1,8]");
  rebuild_node_def(DT_COMPLEX128, DT_INT8);
  INFER_OK(op, "[4,5]", "[d0_0,d0_1,16]");

  // Bitcasting from a POD or quantized datatype is not allowed.
  rebuild_node_def(DT_STRING, DT_INT32);
  INFER_ERROR("one of the type sizes is zero", op, "[1,2,3]");
  rebuild_node_def(DT_INT32, DT_STRING);
  INFER_ERROR("one of the type sizes is zero", op, "[1,2,3]");
}

TEST(ArrayOpsTest, Squeeze_ShapeFn) {
  ShapeInferenceTestOp op("Squeeze");

  auto rebuild_node_def = [&op](const std::vector<int32>& squeeze_dims) {
    TF_ASSERT_OK(NodeDefBuilder("test", "Squeeze")
                     .Input("input", 0, DT_FLOAT)
                     .Attr("squeeze_dims", squeeze_dims)
                     .Finalize(&op.node_def));
  };

  // Default squeeze_dims = []
  rebuild_node_def({});

  // No valid shape provided, so output is unknown.
  INFER_OK(op, "?", "?");

  INFER_OK(op, "[1,4,1,5,1]", "[d0_1,d0_3]");

  // Squeezing all dimensions, but see some unknown values.
  INFER_OK(op, "[1,?,1,?,1]", "?");

  // Test simple squeeze of an explicit dimension
  rebuild_node_def({1});
  INFER_OK(op, "[4,1,5]", "[d0_0,d0_2]");
  // Squeezing unknown dim explicitly, assumes it's 1 at runtime.
  INFER_OK(op, "[4,?,5]", "[d0_0,d0_2]");

  // Attempt to squeeze non-one dimension
  INFER_ERROR("Can not squeeze dim[1]", op, "[4,6,5]");

  // Squeeze multiple dimensions
  rebuild_node_def({1, 2});
  INFER_OK(op, "[4,1,1,5]", "[d0_0,d0_3]");
  rebuild_node_def({1, -2});
  INFER_OK(op, "[4,1,1,5]", "[d0_0,d0_3]");

  // Negative squeeze dim
  rebuild_node_def({-2});
  INFER_OK(op, "[4,1,5]", "[d0_0,d0_2]");

  // Test validation of squeeze dimensions
  rebuild_node_def({-4});
  INFER_ERROR("not in [-3,3)", op, "[1,2,3]");
  rebuild_node_def({3});
  INFER_ERROR("not in [-3,3)", op, "[1,2,3]");
}

TEST(ArrayOpsTest, ReverseSequence_ShapeFn) {
  ShapeInferenceTestOp op("ReverseSequence");
  auto rebuild_node_def = [&op](const int32 seq_dim, const int32 batch_dim) {
    TF_ASSERT_OK(NodeDefBuilder("test", "ReverseSequence")
                     .Input("input", 0, DT_FLOAT)
                     .Input("seq_lengths", 1, DT_INT64)
                     .Attr("seq_dim", seq_dim)
                     .Attr("batch_dim", batch_dim)
                     .Finalize(&op.node_def));
  };

  rebuild_node_def(1, 2);
  // No valid shape provided, so output is unknown.
  INFER_OK(op, "?;[10]", "?");

  // Bad rank for seq_lengths
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[10,10]");

  // Validate seq_dim and batch_dim
  rebuild_node_def(1, 4);
  INFER_ERROR("batch_dim must be < input rank", op, "[1,2,3];[3]");
  rebuild_node_def(4, 1);
  INFER_ERROR("seq_dim must be < input rank", op, "[1,2,3];[3]");

  rebuild_node_def(1, 2);
  INFER_OK(op, "[1,2,3];[3]", "[d0_0,d0_1,d0_2]");
  // Resolves uncertainty on batch dimension by merging.
  INFER_OK(op, "[1,2,?];[3]", "[d0_0,d0_1,d1_0]");
  INFER_OK(op, "[1,2,3];[?]", "[d0_0,d0_1,d0_2]");
}

TEST(ArrayOpsTest, Split_ShapeFn) {
  ShapeInferenceTestOp op("Split");
  op.input_tensors.resize(2);

  // No value for split_dim and no input.
  TF_ASSERT_OK(NodeDefBuilder("test", "Split")
                   .Input("split_dim", 0, DT_INT32)
                   .Input("value", 1, DT_FLOAT)
                   .Attr("num_split", 2)
                   .Finalize(&op.node_def));
  INFER_OK(op, "?;?", "?;?");
  // If the rank is known, we know the rank of each output.
  INFER_OK(op, "?;[?,?]", "[?,?];[?,?]");

  // split_dim is known.
  Tensor split_dim = test::AsTensor<int32>({1, 2});
  op.input_tensors[0] = &split_dim;
  INFER_ERROR("Input must be scalar but has rank 1", op, "[?];[?,?]");
  split_dim = test::AsScalar<int32>(1);
  INFER_OK(op, "?;?", "?;?");
  INFER_OK(op, "?;[?,?]", "[d1_0,?];[d1_0,?]");
  INFER_OK(op, "?;[1,4]", "[d1_0,2];[d1_0,2]");
  INFER_OK(op, "?;[1,?]", "[d1_0,?];[d1_0,?]");
  INFER_ERROR("Dimension size must be evenly divisible by 2 but is 5", op,
              "?;[1,5]");
}

TEST(ArrayOpsTest, Tile_ShapeFn) {
  ShapeInferenceTestOp op("Tile");
  op.input_tensors.resize(2);

  // No value for split_dim and no input.
  TF_ASSERT_OK(NodeDefBuilder("test", "Tile")
                   .Input("input", 0, DT_FLOAT)
                   .Input("multiples", 1, DT_INT32)
                   .Finalize(&op.node_def));

  // If both are unknown, output is unknown.
  INFER_OK(op, "?;?", "?");

  // If multiples rank is unknown but input is, output rank is known.
  INFER_OK(op, "[2,3,1,4];?", "[?,?,?,?]");

  // Bad rank for 'multiples'
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[2,3,1,4];[4,1]");

  // No multiples tensor available, but output rank is known from multiples.
  INFER_OK(op, "?;[4]", "[?,?,?,?]");

  // Test a tile of a 4D input.
  Tensor multiples = test::AsTensor<int32>({2, 3, 4, 5});
  op.input_tensors[1] = &multiples;
  INFER_OK(op, "[2,3,1,4];[4]", "[4,9,4,20]");
  // Test 64-bit tensor type
  multiples = test::AsTensor<int64>({2, 3, 4, 5});
  INFER_OK(op, "[2,3,1,4];[4]", "[4,9,4,20]");
}

TEST(ArrayOpsTest, EditDistance_ShapeFn) {
  ShapeInferenceTestOp op("EditDistance");
  op.input_tensors.resize(6);

  // If the shape tensors are not available, the output shape is unknown.
  INFER_OK(op, "[?,?];[?];[4];[?,?];[?];[4]", "?");

  Tensor hypothesis_shape = test::AsTensor<int64>({2, 30, 4, 50});
  op.input_tensors[2] = &hypothesis_shape;
  Tensor truth_shape = test::AsTensor<int64>({20, 3, 40, 5});
  op.input_tensors[5] = &truth_shape;
  INFER_OK(op, "[?,?];[?];[4];[?,?];[?];[4]", "[20,30,40]");

  // Shape elements don't match
  hypothesis_shape = test::AsTensor<int64>({2});
  op.input_tensors[2] = &hypothesis_shape;
  INFER_ERROR("Num elements of hypothesis_shape does not match truth_shape", op,
              "[?,?];[?];[1];[?,?];[?];[4]");
}

TEST(ArrayOpsTest, OneHot_ShapeFn) {
  ShapeInferenceTestOp op("OneHot");
  op.input_tensors.resize(4);
  auto set_axis = [&op](int axis) {
    TF_ASSERT_OK(NodeDefBuilder("test", "OneHot")
                     .Input("indices", 0, DT_FLOAT)
                     .Input("depth", 1, DT_INT32)
                     .Input("on_value", 2, DT_FLOAT)
                     .Input("off_value", 3, DT_FLOAT)
                     .Attr("axis", axis)
                     .Finalize(&op.node_def));
  };

  // Invalid axis value.
  set_axis(-2);
  INFER_ERROR("axis must be >= -1", op, "?;?;?;?");
  set_axis(1);

  // If indices shape is unknown, we return an unknown shape.
  INFER_OK(op, "?;[];?;?", "?");

  // Depth must be scalar.
  Tensor depth = test::AsTensor<int32>({1, 2});
  op.input_tensors[1] = &depth;
  INFER_ERROR("Input must be scalar but has rank 1", op, "?;[2];?;?");

  // Full information is available.
  depth = test::AsScalar<int32>(2);
  INFER_OK(op, "[1,3,4];[];?;?", "[d0_0,2,d0_1,d0_2]");
  set_axis(-1);
  INFER_OK(op, "[1,3,4];[];?;?", "[d0_0,d0_1,d0_2,2]");
}

TEST(ArrayOpsTest, ExtractImagePatchesShapeTest) {
  ShapeInferenceTestOp op("ExtractImagePatches");
  auto set_op = [&op](const std::vector<int32>& ksizes,
                      const std::vector<int32>& strides,
                      const std::vector<int32>& rates, const string& padding) {
    TF_ASSERT_OK(NodeDefBuilder("test", "ExtractImagePatches")
                     .Input("input", 0, DT_FLOAT)
                     .Attr("ksizes", ksizes)
                     .Attr("strides", strides)
                     .Attr("rates", rates)
                     .Attr("padding", padding)
                     .Finalize(&op.node_def));
  };

  // Just tests that the ksize calculation with rates works.  Most of
  // the other code is boilerplate that is tested by a variety of
  // other ops.
  //
  // ksizes is 2x2.  rate rows and cols is 2, so ksize_rows and
  // cols are changed to be 2 + (2 - 1) = 3.  7x7 input with 3x3
  // filter and 1x1 stride gives a 5x5 output.
  set_op({1, 2, 2, 1}, {1, 1, 1, 1}, {1, 2, 2, 1}, "VALID");
  INFER_OK(op, "[1,7,7,2]", "[d0_0,5,5,8]");
  // With ksizes as 1x1, the output depth is now exactly the last value of the
  // input and output spatial is reduced as well.
  set_op({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 2, 2, 1}, "VALID");
  INFER_OK(op, "[1,7,7,2]", "[d0_0,7,7,d0_3]");

  // Bad ksize rank
  set_op({1, 2, 2, 1, 1}, {1, 1, 1, 1}, {1, 2, 2, 1}, "VALID");
  INFER_ERROR(
      "ExtractImagePatches requires the ksizes attribute to contain 4 values, "
      "but got: 5",
      op, "[1,7,7,2]");
}

TEST(ArrayOpsTest, SpaceToBatch_ShapeFn) {
  ShapeInferenceTestOp op("SpaceToBatch");
  op.input_tensors.resize(2);
  TF_ASSERT_OK(NodeDefBuilder("test", "SpaceToBatch")
                   .Input("input", 0, DT_FLOAT)
                   .Input("paddings", 1, DT_INT32)
                   .Attr("block_size", 2)
                   .Finalize(&op.node_def));

  // Paddings not known, but batch size can be computed.
  INFER_OK(op, "[1,10,10,3];[2,2]", "[4,?,?,d0_3]");

  // Unknown paddings means width and height.
  INFER_OK(op, "[1,10,10,3];?", "[4,?,?,d0_3]");

  // Paddings not correct shape
  INFER_ERROR("rank", op, "[1,10,10,3];[4]");
  INFER_ERROR("3 and 2", op, "[1,10,10,3];[2,3]");

  Tensor paddings = test::AsTensor<int32>({4, 2, 2, 4}, {{2, 2}});
  op.input_tensors[1] = &paddings;
  INFER_OK(op, "[1,10,10,3];[2,2]", "[4,8,8,d0_3]");
  paddings = test::AsTensor<int64>({4, 2, 2, 4}, {{2, 2}});
  INFER_OK(op, "[1,10,10,3];[2,2]", "[4,8,8,d0_3]");

  // Bad paddings values
  paddings = test::AsTensor<int32>({1, 2, 3, 4}, {{2, 2}});
  op.input_tensors[1] = &paddings;
  INFER_ERROR("Dimension size must be evenly divisible by 2 but is 13", op,
              "[1,10,10,3];[2,2]");

  // Negative paddsings
  paddings = test::AsTensor<int32>({1, -2, 3, 4}, {{2, 2}});
  op.input_tensors[1] = &paddings;
  INFER_ERROR("cannot be negative", op, "[1,10,10,3];[2,2]");
}

TEST(ArrayOpsTest, SpaceToBatchND_ShapeFn) {
  ShapeInferenceTestOp op("SpaceToBatchND");
  op.input_tensors.resize(3);
  TF_ASSERT_OK(NodeDefBuilder("test", "SpaceToBatchND")
                   .Input("input", 0, DT_FLOAT)
                   .Input("block_shape", 1, DT_INT32)
                   .Input("paddings", 2, DT_INT32)
                   .Finalize(&op.node_def));

  // Verify that input shape and paddings shape can be unknown.
  INFER_OK(op, "?;[2];?", "?");

  // Only number of input dimensions is known.
  INFER_OK(op, "[?,?,?,?];[2];?", "[?,?,?,d0_3]");

  // Dimensions are partially known.
  INFER_OK(op, "[?,?,?,2];[2];?", "[?,?,?,d0_3]");

  {
    // Dimensions are partially known, block_shape known.
    Tensor block_shape = test::AsTensor<int32>({2, 3});
    op.input_tensors[1] = &block_shape;
    INFER_OK(op, "[3,?,?,2];[2];?", "[18,?,?,d0_3]");

    // Dimensions are partially known, block_shape and paddings known.
    {
      Tensor paddings = test::AsTensor<int32>({1, 1, 0, 1}, {{2, 2}});
      op.input_tensors[2] = &paddings;
      INFER_OK(op, "[3,?,2,2];[2];[2,2]", "[18,?,1,d0_3]");
      op.input_tensors[2] = nullptr;
    }

    // Dimensions are fully known, block_shape and paddings are known.
    {
      Tensor paddings = test::AsTensor<int32>({1, 1, 0, 0}, {{2, 2}});
      op.input_tensors[2] = &paddings;
      INFER_OK(op, "[3,2,3,2];[2];[2,2]", "[18,2,1,d0_3]");
      op.input_tensors[2] = nullptr;
    }

    op.input_tensors[1] = nullptr;
  }

  INFER_ERROR("block_shape must have rank 1", op, "?;[1,1];?");
  INFER_ERROR("block_shape must have known size", op, "?;[?];?");

  {
    Tensor block_shape = test::AsTensor<int32>({0, 2});
    op.input_tensors[1] = &block_shape;
    INFER_ERROR("block_shape must be positive", op, "[1,2,2];[2];[2,2]");
    op.input_tensors[1] = nullptr;
  }

  {
    Tensor block_shape = test::AsTensor<int32>({1, 1});
    op.input_tensors[1] = &block_shape;
    Tensor paddings = test::AsTensor<int32>({0, -1, 0, 0}, {{2, 2}});
    op.input_tensors[2] = &paddings;
    INFER_ERROR("paddings cannot be negative", op, "[1,2,2];[2];[2,2]");
    op.input_tensors[1] = nullptr;
    op.input_tensors[2] = nullptr;
  }

  {
    Tensor block_shape = test::AsTensor<int32>({3, 3});
    op.input_tensors[1] = &block_shape;
    Tensor paddings = test::AsTensor<int32>({0, 0, 0, 0}, {{2, 2}});
    op.input_tensors[2] = &paddings;
    INFER_ERROR("divisible", op, "[1,2,3,1];[2];[2,2]");
    op.input_tensors[1] = nullptr;
    op.input_tensors[2] = nullptr;
  }

  INFER_ERROR("rank", op, "[1,3,3,1];[2];[1]");
  INFER_ERROR("shape", op, "[1,3,3,1];[2];[1,2]");
}

TEST(ArrayOpsTest, BatchToSpace_ShapeFn) {
  ShapeInferenceTestOp op("BatchToSpace");
  op.input_tensors.resize(2);
  TF_ASSERT_OK(NodeDefBuilder("test", "BatchToSpace")
                   .Input("input", 0, DT_FLOAT)
                   .Input("crops", 1, DT_INT32)
                   .Attr("block_size", 2)
                   .Finalize(&op.node_def));

  // croppings not known, but batch size can be computed.
  INFER_OK(op, "[4,8,8,3];[2,2]", "[1,?,?,d0_3]");

  // block_size not compatible with batch size
  INFER_ERROR("Dimension size must be evenly divisible by", op,
              "[5,8,8,3];[2,2]");

  // Unknown croppings means unknown width and height.
  INFER_OK(op, "[4,8,8,3];?", "[1,?,?,d0_3]");

  // croppings not correct shape
  INFER_ERROR("rank", op, "[4,8,8,3];[4]");
  INFER_ERROR("3 and 2", op, "[4,8,8,3];[2,3]");

  Tensor croppings = test::AsTensor<int64>({4, 2, 2, 4}, {{2, 2}});
  op.input_tensors[1] = &croppings;
  INFER_OK(op, "[4,8,8,3];[2,2]", "[1,10,10,d0_3]");

  // Bad croppings values
  croppings = test::AsTensor<int32>({100, 2, 3, 4}, {{2, 2}});
  op.input_tensors[1] = &croppings;
  INFER_ERROR("Negative dimension size caused by subtracting", op,
              "[4,8,8,3];[2,2]");
  croppings = test::AsTensor<int32>({1, 2, 3, 400}, {{2, 2}});
  op.input_tensors[1] = &croppings;
  INFER_ERROR("Negative dimension size caused by subtracting", op,
              "[4,8,8,3];[2,2]");

  // Negative paddsings
  croppings = test::AsTensor<int32>({1, -2, 3, 4}, {{2, 2}});
  op.input_tensors[1] = &croppings;
  INFER_ERROR("cannot be negative", op, "[4,8,8,3];[2,2]");
}

TEST(ArrayOpsTest, BatchToSpaceND_ShapeFn) {
  ShapeInferenceTestOp op("BatchToSpaceND");
  op.input_tensors.resize(3);
  TF_ASSERT_OK(NodeDefBuilder("test", "BatchToSpaceND")
                   .Input("input", 0, DT_FLOAT)
                   .Input("block_shape", 1, DT_INT32)
                   .Input("crops", 2, DT_INT32)
                   .Finalize(&op.node_def));

  // Verify that input shape and crops shape can be unknown.
  INFER_OK(op, "?;[2];?", "?");

  // Only number of input dimensions is known.
  INFER_OK(op, "[?,?,?,?];[2];?", "[?,?,?,d0_3]");

  {
    // Dimensions are partially known, block_shape known.
    Tensor block_shape = test::AsTensor<int32>({2, 3});
    op.input_tensors[1] = &block_shape;
    INFER_OK(op, "[?,?,?,2];[2];?", "[?,?,?,d0_3]");

    INFER_OK(op, "[18,?,?,2];[2];?", "[3,?,?,d0_3]");

    // Dimensions are partially known, block_shape and crops known.
    {
      Tensor crops = test::AsTensor<int32>({1, 1, 0, 1}, {{2, 2}});
      op.input_tensors[2] = &crops;
      INFER_OK(op, "[18,?,2,2];[2];[2,2]", "[3,?,5,d0_3]");
      op.input_tensors[2] = nullptr;
    }

    // Dimensions are fully known, block_shape and crops are known.
    {
      Tensor crops = test::AsTensor<int32>({1, 1, 0, 0}, {{2, 2}});
      op.input_tensors[2] = &crops;
      INFER_OK(op, "[18,2,1,2];[2];[2,2]", "[3,2,3,d0_3]");
      op.input_tensors[2] = nullptr;
    }

    op.input_tensors[1] = nullptr;
  }

  INFER_ERROR("block_shape must have rank 1", op, "?;[1,1];?");
  INFER_ERROR("block_shape must have known size", op, "?;[?];?");
  INFER_ERROR("rank", op, "[2,2];[2];[2,2]");
  INFER_ERROR("rank", op, "[2,2,3];[3];[3,2]");

  {
    Tensor block_shape = test::AsTensor<int32>({0, 2});
    op.input_tensors[1] = &block_shape;
    INFER_ERROR("block_shape must be positive", op, "[1,2,2];[2];[2,2]");
    op.input_tensors[1] = nullptr;
  }

  {
    Tensor block_shape = test::AsTensor<int32>({1, 1});
    op.input_tensors[1] = &block_shape;
    Tensor paddings = test::AsTensor<int32>({0, -1, 0, 0}, {{2, 2}});
    op.input_tensors[2] = &paddings;
    INFER_ERROR("crops cannot be negative", op, "[1,2,2];[2];[2,2]");
    op.input_tensors[1] = nullptr;
    op.input_tensors[2] = nullptr;
  }

  // The amount to crop exceeds the padded size.
  {
    Tensor block_shape = test::AsTensor<int32>({2, 2});
    op.input_tensors[1] = &block_shape;
    Tensor crops = test::AsTensor<int32>({3, 2, 0, 0}, {{2, 2}});
    op.input_tensors[2] = &crops;
    INFER_ERROR("Negative", op, "[4,2,3,1];[2];[2,2]");
    op.input_tensors[1] = nullptr;
    op.input_tensors[2] = nullptr;
  }

  // The batch size is not divisible by the product of the block_shape.
  {
    Tensor block_shape = test::AsTensor<int32>({2, 3});
    op.input_tensors[1] = &block_shape;
    INFER_ERROR("divisible", op, "[3,1,1,1];[2];[2,2]");
    op.input_tensors[1] = nullptr;
  }
}

TEST(ArrayOpsTest, SpaceToDepth_ShapeFn) {
  ShapeInferenceTestOp op("SpaceToDepth");
  TF_ASSERT_OK(NodeDefBuilder("test", "SpaceToDepth")
                   .Input("input", 0, DT_FLOAT)
                   .Attr("block_size", 2)
                   .Finalize(&op.node_def));

  INFER_OK(op, "[1,2,4,4]", "[d0_0,1,2,16]");

  // block_size not compatible with space
  INFER_ERROR("Dimension size must be evenly divisible by 2 but is 3", op,
              "[1,3,8,4]");
  INFER_ERROR("Dimension size must be evenly divisible by 2 but is 5", op,
              "[1,2,5,4]");

  // Unknown depth --> Unknown depth.
  INFER_OK(op, "[1,2,4,?]", "[d0_0,1,2,?]");
}

TEST(ArrayOpsTest, DepthToSpace_ShapeFn) {
  ShapeInferenceTestOp op("DepthToSpace");
  TF_ASSERT_OK(NodeDefBuilder("test", "DepthToSpace")
                   .Input("input", 0, DT_FLOAT)
                   .Attr("block_size", 2)
                   .Finalize(&op.node_def));

  INFER_OK(op, "[1,1,2,16]", "[d0_0,2,4,4]");

  // Bad depth
  INFER_ERROR("Dimension size must be evenly divisible by 4 but is 15", op,
              "[1,1,2,15]");

  // Unknown depth --> Unknown depth.
  INFER_OK(op, "[1,2,4,?]", "[d0_0,4,8,?]");

  // Check another block size.
  TF_ASSERT_OK(NodeDefBuilder("test", "DepthToSpace")
                   .Input("input", 0, DT_FLOAT)
                   .Attr("block_size", 10)
                   .Finalize(&op.node_def));
  INFER_OK(op, "[1,1,2,200]", "[d0_0,10,20,2]");
}

TEST(ArrayOpsTest, Slice_ShapeFn) {
  ShapeInferenceTestOp op("Slice");
  TF_ASSERT_OK(NodeDefBuilder("test", "Slice")
                   .Input("input", 0, DT_FLOAT)
                   .Input("begin", 1, DT_INT64)
                   .Input("sizes", 2, DT_INT64)
                   .Finalize(&op.node_def));

  // Known rank of input and shape of begin/sizes, but unknown values.
  // The best we know is the rank of the output.
  INFER_OK(op, "[2,3,4,5];[4];[4]", "[?,?,?,?]");

  // Unknown shape of begin/sizes, we still know the rank.
  INFER_OK(op, "[2,3,4,5];[?];[?]", "[?,?,?,?]");
  // Unknown all around
  INFER_OK(op, "?;[?];[?]", "?");
  // Can infer based on begin
  INFER_OK(op, "?;[4];[?]", "[?,?,?,?]");

  // Bad rank of begin, sizes
  INFER_ERROR("must be rank 1", op, "[2,3,4,5];[2,3];[3]");
  INFER_ERROR("must be rank 1", op, "[2,3,4,5];[2];[3,4]");
  // Length of begin doesn't match input rank
  INFER_ERROR("must be rank 2", op, "[2,3,4,5];[2];[2]");

  // Tests with known values.
  op.input_tensors.resize(3);
  Tensor begin = test::AsTensor<int32>({0, 1, 2, 1});
  Tensor sizes = test::AsTensor<int32>({1, 2, 1, 3});
  op.input_tensors[1] = &begin;
  op.input_tensors[2] = &sizes;
  INFER_OK(op, "[2,3,4,5];[4];[4]", "[1,2,1,3]");

  // -1 in sizes means "get the rest"
  sizes = test::AsTensor<int32>({-1, -1, 1, -1});
  INFER_OK(op, "[2,3,4,5];[4];[4]", "[d0_0,2,1,4]");

  begin = test::AsTensor<int32>({0, 1, 2, 6});
  sizes = test::AsTensor<int32>({-1, -1, -1, -1});
  INFER_ERROR("Negative dimension size", op, "[2,3,4,5];[4];[4]");

  begin = test::AsTensor<int32>({0, 1, 2, 5});
  sizes = test::AsTensor<int32>({-1, -1, -1, -2});
  INFER_ERROR("cannot be < -1", op, "[2,3,4,5];[4];[4]");
}

TEST(ArrayOpsTest, StridedSliceGrad_ShapeFn) {
  ShapeInferenceTestOp op("StridedSliceGrad");
  op.input_tensors.resize(5);
  INFER_OK(op, "?;?;?;?;?", "?");
  INFER_OK(op, "[?];?;?;?;?", "?");
  INFER_OK(op, "[4];?;?;?;?", "[?,?,?,?]");

  Tensor in_t = test::AsTensor<int32>({1, 2, 3, 4});
  op.input_tensors[0] = &in_t;
  INFER_OK(op, "[4];?;?;?;?", "[1,2,3,4]");
}

TEST(ArrayOpsTest, UnchangedWithQuantizationScalars_ShapeFn) {
  for (const char* op_name : {"Dequantize", "FakeQuantWithMinMaxVars"}) {
    ShapeInferenceTestOp op(op_name);

    INFER_OK(op, "?;?;?", "in0");
    INFER_OK(op, "[1,?,3];[];[]", "in0");

    // Rank check scalars.
    INFER_ERROR("be rank 0", op, "[1,?,3];[1];[]");
    INFER_ERROR("be rank 0", op, "[1,?,3];[];[1]");
  }
}

TEST(ArrayOpsTest, FakeQuantWithMinMaxVarsPerChannel) {
  ShapeInferenceTestOp op("FakeQuantWithMinMaxVarsPerChannel");

  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "[?];?;?", "in0");
  INFER_OK(op, "[1,?,3];[3];[3]", "in0");
  INFER_OK(op, "[3];[3];[3]", "in0");

  // Rank check vectors.
  INFER_ERROR("be rank 1", op, "[1,?,3];[1];[]");
  INFER_ERROR("be rank 1", op, "[1,?,3];[];[1]");

  // Vectors must match each other, and match last dim of input.
  INFER_ERROR("must be equal", op, "[1,?,3];[2];[?]");
  INFER_ERROR("must be equal", op, "[1,?,3];[?];[2]");
  INFER_ERROR("must be equal", op, "[1,?,?];[1];[2]");
  INFER_ERROR("must be equal", op, "[5];[4];[?]");
}

TEST(ArrayOpsTest, FakeQuantWithMinMaxVarsPerChannelGradient) {
  ShapeInferenceTestOp op("FakeQuantWithMinMaxVarsPerChannelGradient");

  INFER_OK(op, "?;?;?;?", "?;[?];[?]");
  INFER_OK(op, "[3];[3];[3];[3]", "in0;in3;in3");
  INFER_OK(op, "[1,3];[1,3];[3];[3]", "in0;in3;in3");
  INFER_OK(op, "[1,2,3,4];[1,2,3,4];[4];[4]", "in0;in3;in3");

  // Rank check vectors.
  INFER_ERROR("be equal rank", op, "[1,?,3];[1,?,3];[3];[]");
  INFER_ERROR("be rank 1", op, "[1,?,3];[1,?,3];[];[3]");
  INFER_ERROR("be at least rank 1", op, "[];[];[1];[1]");
  INFER_ERROR("be at most rank 4", op, "[1,2,3,4,5];[1,2,3,4,5];[1];[1]");

  // Vectors must match each other, and match last dim of input.
  INFER_ERROR("must be equal", op, "[1,3];[1,3];[2];[3]");
  INFER_ERROR("must be equal", op, "[1,3];[1,3];[3];[2]");
}

TEST(ArrayOpsTest, QuantizedConcat_ShapeFn) {
  ShapeInferenceTestOp op("QuantizedConcat");
  auto set_n = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> src_list;
    std::vector<NodeDefBuilder::NodeOut> limit_list;
    for (int i = 0; i < n; ++i) {
      src_list.emplace_back("a", 0, DT_QUINT8);
      limit_list.emplace_back("b", 0, DT_FLOAT);
    }
    TF_ASSERT_OK(NodeDefBuilder("test", "QuantizedConcat")
                     .Input({"concat_dim", 0, DT_INT32})
                     .Input(src_list)
                     .Input(limit_list)
                     .Input(limit_list)
                     .Attr("N", n)
                     .Finalize(&op.node_def));
  };

  // Confirm dimension[0] of the input (the concat_dim) is a scalar.
  set_n(1);
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[1];?;?;?");

  // Last 2*<N> are all scalars.
  set_n(2);
  INFER_ERROR("must be rank 0", op, "[];?;?;?;?;?;[1]");
  INFER_ERROR("must be rank 0", op, "[];?;?;?;?;[1];?");
  INFER_ERROR("must be rank 0", op, "[];?;?;?;[1];?;?");
  INFER_ERROR("must be rank 0", op, "[];?;?;[1];?;?;?");

  // First is concat dim; next N must be compatible for concat.
  set_n(2);
  INFER_ERROR("must be rank 2", op, "[];[1,2];[1,2,3];?;?;?;?");
  INFER_OK(op, "[];[1,2];[1,3];?;?;?;?", "[?,?];[];[]");

  // Test when the concat_dim tensor is known. The concatenated dimension is
  // summed across all input tensors, and other dimensions are merged.
  Tensor concat_dim_t;
  op.input_tensors.push_back(&concat_dim_t);
  set_n(2);
  concat_dim_t = test::AsScalar(0);  // Sum dim 0, merge the other two dims.
  INFER_OK(op, "[];[100,2,?];[10,?,3];?;?;?;?", "[110,d1_1,d2_2];[];[]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 5 and 3", op,
              "[];[100,2,5];[10,?,3];?;?;?;?");
  // Note that other cases of concat are covered in the Concat tests.
}

}  // end namespace tensorflow
