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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(NNOpsTest, TopK_ShapeFn) {
  ShapeInferenceTestOp op("TopK");
  auto set_k = [&op](int k) {
    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input({{"a", 0, DT_FLOAT}})
                     .Attr("k", k)
                     .Finalize(&op.node_def));
  };

  set_k(20);
  // With known input, each output is an unknown shape.
  INFER_OK(op, "?", "?;?");
  // With vector input, each output is [k].
  INFER_OK(op, "[20]", "[20];[20]");
  INFER_OK(op, "[21]", "[20];[20]");

  // With input rank 3, each output is the two first 2 dims of input, plus k.
  INFER_OK(op, "[1,?,21]", "[d0_0,d0_1,20];[d0_0,d0_1,20]");
  // With input rank 4, each output is the two first 3 dims of input, plus k.
  INFER_OK(op, "[1,?,21,?]", "[d0_0,d0_1,d0_2,20];[d0_0,d0_1,d0_2,20]");

  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 1", op, "[1]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 4", op,
              "[1,2,3,4]");
  set_k(-1);
  INFER_ERROR("Need k >= 0, got -1", op, "[1,2,3,4]");
}

TEST(NNOpsTest, TopKV2_ShapeFn) {
  ShapeInferenceTestOp op("TopKV2");
  op.input_tensors.resize(2);

  Tensor k_t;
  op.input_tensors[1] = &k_t;

  k_t = test::AsScalar<int32>(20);
  // With known input, each output is an unknown shape.
  INFER_OK(op, "?;[]", "?;?");
  // With vector input, each output is [k].
  INFER_OK(op, "[20];[]", "[20];[20]");

  // With input rank 3, each output is the two first 2 dims of input, plus k.
  INFER_OK(op, "[1,?,21];[]", "[d0_0,d0_1,20];[d0_0,d0_1,20]");
  // With input rank 4, each output is the two first 3 dims of input, plus k.
  INFER_OK(op, "[1,?,21,?];[]", "[d0_0,d0_1,d0_2,20];[d0_0,d0_1,d0_2,20]");

  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 1", op,
              "[1];[]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 4", op,
              "[1,2,3,4];[]");
  k_t = test::AsScalar<int32>(-1);
  INFER_ERROR(
      "Dimension size, given by scalar input 1, must be non-negative but is -1",
      op, "[1,2,3,4];[]");
}

TEST(NNOpsTest, InputTensorShapeOrUnknown2D_ShapeFn) {
  typedef std::pair<const char*, int> NameAndInputIndex;
  for (const auto& p :
       {NameAndInputIndex("AvgPoolGrad", 0),
        NameAndInputIndex("Conv2DBackpropInput", 0),
        NameAndInputIndex("Conv2DBackpropFilter", 1),
        NameAndInputIndex("DepthwiseConv2dNativeBackpropInput", 0),
        NameAndInputIndex("DepthwiseConv2dNativeBackpropFilter", 1)}) {
    ShapeInferenceTestOp op(p.first);
    op.input_tensors.resize(2);

    // Conv and Depthwise conv have three inputs.
    string extra_shapes = (op.name == "AvgPoolGrad" ? "" : ";?");

    // When the input tensor is not known, the output is 4 unknown dims.
    INFER_OK(op, "?;?" + extra_shapes, "[?,?,?,?]");
    INFER_OK(op, "[4];?" + extra_shapes, "[?,?,?,?]");

    // When input tensor is known, its values determine output shape.
    std::vector<int32> shape{1, 2, 3, 4};
    Tensor shape_t = test::AsTensor<int32>(shape);
    op.input_tensors[p.second] = &shape_t;
    INFER_OK(op, "[4];?" + extra_shapes, "[1,2,3,4]");
  }
}

TEST(NNOpsTest, InputTensorShapeOrUnknown3D_ShapeFn) {
  typedef std::pair<const char*, int> NameAndInputIndex;
  for (const auto& p : {NameAndInputIndex("AvgPool3DGrad", 0),
                        NameAndInputIndex("Conv3DBackpropInputV2", 0),
                        NameAndInputIndex("Conv3DBackpropFilterV2", 1)}) {
    ShapeInferenceTestOp op(p.first);
    op.input_tensors.resize(2);

    // Conv3D has an extra shape.
    string extra_shapes = (op.name == "AvgPool3DGrad" ? "" : ";?");

    // When the input tensor is not known, the output is 4 unknown dims.
    INFER_OK(op, "?;?" + extra_shapes, "[?,?,?,?,?]");
    INFER_OK(op, "[5];?" + extra_shapes, "[?,?,?,?,?]");

    // When input tensor is known, its values determine output shape.
    std::vector<int32> shape{1, 2, 3, 4, 5};
    Tensor shape_t = test::AsTensor<int32>(shape);
    op.input_tensors[p.second] = &shape_t;
    INFER_OK(op, "[5];?" + extra_shapes, "[1,2,3,4,5]");
  }
}

TEST(NNOpsTest, BatchNormWithGlobalNormalization_ShapeFn) {
  ShapeInferenceTestOp op("BatchNormWithGlobalNormalization");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;?;[1,2,3]");

  // last dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,?,?,d3_0]");
  INFER_OK(op, "?;?;?;?;[1]", "[?,?,?,d4_0]");
  INFER_OK(op, "[1,2,3,4];[4];[4];[4];[4]",
           "[d0_0,d0_1,d0_2,d0_3|d1_0|d2_0|d3_0|d4_0]");
}

TEST(NNOpsTest, BatchNormWithGlobalNormalizationGrad_ShapeFn) {
  ShapeInferenceTestOp op("BatchNormWithGlobalNormalizationGrad");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shapes must be equal rank, but are 4 and 3", op,
              "?;?;?;?;[1,2,3]");

  // The first output comes from the first and last inputs merged together.
  // Other inputs are merged with the last dim of that merge result, and that
  // merged vector dim is the last 4 outputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,?,?,d3_0];[d3_0];[d3_0];[d3_0];[d3_0]");
  INFER_OK(op, "[1,?,3,?];[?];[?];[?];[?,2,?,4]",
           "[d0_0,d4_1,d0_2,d4_3];[d4_3];[d4_3];[d4_3];[d4_3]");
}

TEST(NNOpsTest, Conv3DBackpropInput_ShapeFn) {
  ShapeInferenceTestOp op("Conv3DBackpropInput");

  // Test rank error.
  INFER_ERROR("Shape must be rank 5 but is rank 3", op, "[1,2,3];?;?");

  // input[1] is transferred to output after asserting its rank.
  INFER_OK(op, "?;?;?", "[?,?,?,?,?]");
  INFER_OK(op, "[?,?,?,?,?];?;?", "in0");
  INFER_OK(op, "[?,2,?,4,?];?;?", "in0");
}

TEST(NNOpsTest, Conv3DBackpropFilter_ShapeFn) {
  ShapeInferenceTestOp op("Conv3DBackpropFilter");

  // Test rank error.
  INFER_ERROR("Shape must be rank 5 but is rank 3", op, "?;[1,2,3];?");

  // input[1] is transferred to output after asserting its rank.
  INFER_OK(op, "?;?;?", "[?,?,?,?,?]");
  INFER_OK(op, "?;[?,?,?,?,?];?", "in1");
  INFER_OK(op, "?;[?,2,?,4,?];?", "in1");
}

TEST(NNOpsTest, MaxPool3DGrad_ShapeFn) {
  ShapeInferenceTestOp op("MaxPool3DGrad");

  // Test rank error.
  INFER_ERROR("Shape must be rank 5 but is rank 3", op, "[1,2,3];?;?");

  // input[0] is transferred to output after asserting its rank.
  INFER_OK(op, "?;?;?", "[?,?,?,?,?]");
  INFER_OK(op, "[?,?,?,?,?];?;?", "in0");
  INFER_OK(op, "[?,2,?,4,?];?;?", "in0");
}

TEST(NNOpsTest, LRNGrad_ShapeFn) {
  ShapeInferenceTestOp op("LRNGrad");

  // LRN Grad is a merge of all three inputs, of rank 4.
  INFER_OK(op, "[1,?,?,4];[?,2,?,?];[?,?,3,?]", "[d0_0,d1_1,d2_2,d0_3]");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?");
  INFER_ERROR("Shapes must be equal rank, but are 4 and 3", op, "?;[1,2,3];?");
  INFER_ERROR("Shapes must be equal rank, but are 4 and 3", op, "?;?;[1,2,3]");
}

TEST(NNOpsTest, MaxPoolGrad_ShapeFn) {
  for (const char* op_name : {"MaxPoolGrad", "MaxPoolGradWithArgmax"}) {
    ShapeInferenceTestOp op(op_name);

    // Test rank error.
    INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?");

    // input[0] is transferred to output after asserting its rank.
    INFER_OK(op, "?;?;?", "[?,?,?,?]");
    INFER_OK(op, "[?,?,?,?];?;?", "in0");
    INFER_OK(op, "[?,2,?,4];?;?", "in0");
  }
}

TEST(NNOpsTest, Dilation2DBackpropInput_ShapeFn) {
  ShapeInferenceTestOp op("Dilation2DBackpropInput");

  // input[0] is transferred to output.
  INFER_OK(op, "?;?;?", "in0");
  INFER_OK(op, "?;[?,?,?,?,?];?", "in0");
  INFER_OK(op, "?;[?,2,?,4,?];?", "in0");
}

TEST(NNOpsTest, Dilation2DBackpropFilter_ShapeFn) {
  ShapeInferenceTestOp op("Dilation2DBackpropFilter");

  // input[1] is transferred to output.
  INFER_OK(op, "?;?;?", "in1");
  INFER_OK(op, "?;[?,?,?,?,?];?", "in1");
  INFER_OK(op, "?;[?,2,?,4,?];?", "in1");
}

TEST(NNOpsTest, MergeBothInputs_ShapeFn) {
  for (const char* op_name :
       {"ReluGrad", "Relu6Grad", "EluGrad", "SoftplusGrad", "SoftsignGrad"}) {
    ShapeInferenceTestOp op(op_name);

    INFER_OK(op, "?;?", "in0|in1");
    INFER_OK(op, "?;[1,?,3]", "in1");
    INFER_OK(op, "[1,?,3];?", "in0");
    INFER_OK(op, "[1,?];[?,2]", "[d0_0,d1_1]");
    INFER_ERROR("Dimension 1 in both shapes must be equal, but are 3 and 2", op,
                "[1,3];[?,2]");
  }
}

TEST(NNOpsTest, SoftmaxCrossEntropyWithLogits_ShapeFn) {
  ShapeInferenceTestOp op("SoftmaxCrossEntropyWithLogits");

  // Inputs are [batch_size,N] and [batch_size,N], and outputs are [batch_size]
  // and
  // [batch_size,N].
  INFER_OK(op, "?;?", "[?];[?,?]");
  INFER_OK(op, "[?,?];[?,?]", "[d0_0|d1_0];in0|in1");
  INFER_OK(op, "[1,2];[?,2]", "[d0_0];in0");
  INFER_OK(op, "[1,?];[?,2]", "[d0_0];[d0_0,d0_1|d1_1]");
  INFER_OK(op, "[?,2];[1,2]", "[d1_0];in1");

  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1,?];[2,?]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[1,2,3];?");
  INFER_ERROR("Shapes must be equal rank, but are 2 and 3", op, "?;[1,2,3]");
}

TEST(NNOpsTest, SparseSoftmaxCrossEntropyWithLogits_ShapeFn) {
  ShapeInferenceTestOp op("SparseSoftmaxCrossEntropyWithLogits");

  // Inputs are [batch_size,N] and [batch_size], and outputs are [batch_size]
  // and [batch_size,N].
  INFER_OK(op, "?;?", "[?];[?,?]");
  INFER_OK(op, "[?,?];[?]", "[d0_0|d1_0];[d0_0|d1_0,d0_1]");
  INFER_OK(op, "[1,2];[1]", "[d0_0|d1_0];[d0_0|d1_0,d0_1]");
  INFER_OK(op, "[?,2];[1]", "[d1_0];[d1_0,d0_1]");

  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,?];[2]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[1,2]");
}

TEST(NNOpsTest, InTopK_ShapeFn) {
  ShapeInferenceTestOp op("InTopK");

  // Inputs are [batch_size,N] and [batch_size], and output is [batch_size].
  INFER_OK(op, "?;?", "[?]");
  INFER_OK(op, "[?,?];[?]", "[d0_0|d1_0]");
  INFER_OK(op, "[1,2];[1]", "[d0_0|d1_0]");
  INFER_OK(op, "[?,2];[1]", "[d1_0]");

  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,?];[2]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[1,2]");
}

TEST(NNOpsTest, Dilation2DShapeTest) {
  ShapeInferenceTestOp op("Dilation2D");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& rates, const string& padding) {
    TF_ASSERT_OK(NodeDefBuilder("test", "Dilation2D")
                     .Input("input", 0, DT_FLOAT)
                     .Input("filter", 0, DT_FLOAT)
                     .Attr("strides", strides)
                     .Attr("rates", rates)
                     .Attr("padding", padding)
                     .Finalize(&op.node_def));
  };

  // rate rows and cols is 1, so filter_rows and cols are unchanged.
  // We have a 1x1 filter so the output is still 2x2.
  set_op({1, 1, 1, 1}, {1, 1, 1, 1}, "VALID");
  INFER_OK(op, "[1,2,2,2];[1,1,2]", "[d0_0,2,2,d1_2]");

  // rate rows and cols is 2, so filter_rows and cols are changed to
  // be 2 + (2 - 1) = 3.  7x7 input with 3x3 filter and 1x1 stride
  // gives a 5x5 output.
  set_op({1, 1, 1, 1}, {1, 2, 2, 1}, "VALID");
  INFER_OK(op, "[1,7,7,2];[2,2,2]", "[d0_0,5,5,d1_2]");
}

TEST(NNOpsTest, FractionalPool_ShapeFn) {
  for (const char* op_name : {"FractionalAvgPool", "FractionalMaxPool"}) {
    ShapeInferenceTestOp op(op_name);
    auto set_op = [&op, op_name](const std::vector<float>& pooling_ratio) {
      TF_ASSERT_OK(NodeDefBuilder("test", op_name)
                       .Input("input", 0, DT_FLOAT)
                       .Attr("pooling_ratio", pooling_ratio)
                       .Finalize(&op.node_def));
    };

    set_op(std::vector<float>{2.0, 1, 1 / 1.5, 1 / 2.0});

    // Rank check.
    INFER_ERROR("must be rank 4", op, "[?,?,?]");

    // Unknown inputs.
    INFER_OK(op, "?", "[?,?,?,?];[?];[?]");
    INFER_OK(op, "[?,?,?,?]", "[?,?,?,?];[?];[?]");

    INFER_OK(op, "[10,20,30,40]", "[5,20,45,80];[20];[45]");
    INFER_OK(op, "[?,20,30,40]", "[?,20,45,80];[20];[45]");
    INFER_OK(op, "[10,?,30,40]", "[5,?,45,80];[?];[45]");
    INFER_OK(op, "[10,20,?,40]", "[5,20,?,80];[20];[?]");
    INFER_OK(op, "[10,20,30,?]", "[5,20,45,?];[20];[45]");

    // Wrong number of values for pooling_ratio.
    set_op(std::vector<float>{.5, 1.0, 1.5});
    INFER_ERROR("pooling_ratio field", op, "?");
    set_op(std::vector<float>{1, 2, 3, 4, 5});
    INFER_ERROR("pooling_ratio field", op, "?");

    // Check dim size >= 0.
    set_op(std::vector<float>{-1, 2, 3, 4});
    INFER_ERROR("is negative", op, "[1,2,3,4]");
  }
}

TEST(NNOpsTest, FractionalMaxPoolGrad) {
  ShapeInferenceTestOp op("FractionalMaxPoolGrad");

  // Note that the shape fn only uses input[0] for computation.
  INFER_ERROR("must be rank 4", op, "[?,?,?];?;?;?;?");
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?]");
  INFER_OK(op, "[?,?,3,4];?;?;?;?", "in0");
}

TEST(NNOpsTest, FractionalAvgPoolGrad) {
  ShapeInferenceTestOp op("FractionalAvgPoolGrad");
  op.input_tensors.resize(1);

  // With no input shape tensor, returns unknown of rank 4.
  INFER_OK(op, "?;?;?;?", "[?,?,?,?]");

  // When input tensor is known, its values determine output shape.
  std::vector<int32> shape{1, 2, 3, 4};
  Tensor shape_t = test::AsTensor<int32>(shape);
  op.input_tensors[0] = &shape_t;
  INFER_OK(op, "[5];?;?;?", "[1,2,3,4]");
}

}  // end namespace tensorflow
