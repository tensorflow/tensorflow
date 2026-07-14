/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace tensorflow {

namespace {

constexpr int32_t kInt8Min = std::numeric_limits<int8_t>::min();
constexpr int32_t kInt8Max = std::numeric_limits<int8_t>::max();
constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();
constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();

}  // namespace

TEST(UniformQuantizedOpsTest, UniformQuantizedDotShapeInference) {
  ShapeInferenceTestOp op("UniformQuantizedDot");
  INFER_OK(op, "[4,2];[2,3];[];[];[];[];[];[]", "[d0_0,d1_1]");
  INFER_OK(op, "[4,2];[2,3];[];[];[3];[3];[];[]", "[d0_0,d1_1]");
  INFER_OK(op, "[4,2];[2,3];[];[];[3];[3];[3];[3]", "[d0_0,d1_1]");

  // Inner dim does not match.
  INFER_ERROR("", op, "[4,2];[6,3];[];[];[];[];[];[]");
  // lhs scales and zero_points must be scalar tensors.
  INFER_ERROR("", op, "[4,2];[2,3];[4];[4];[];[];[];[]");
  // scales and zero_points must have same rank.
  INFER_ERROR("scales and zero_points must have same rank.", op,
              "[4,2];[2,3];[];[];[3];[];[];[]");
  // If rhs scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs.dim_size[1].
  INFER_ERROR("", op, "[4,2];[2,3];[];[];[6];[6];[];[]");
  // If output scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs.dim_size[1].
  INFER_ERROR("", op, "[4,2];[2,3];[];[];[];[];[6];[6]");
}

TEST(UniformQuantizedOpsTest, UniformQuantizedDotHybridShapeInference) {
  ShapeInferenceTestOp op("UniformQuantizedDotHybrid");
  INFER_OK(op, "[4,2];[2,3];[];[]", "[d0_0,d1_1]");
  INFER_OK(op, "[4,2];[2,3];[3];[3]", "[d0_0,d1_1]");

  // Inner dim does not match.
  INFER_ERROR("", op, "[4,2];[6,3];[];[]");
  // scales and zero_points must have same rank.
  INFER_ERROR("scales and zero_points must have same rank.", op,
              "[4,2];[2,3];[3];[]");
  // If rhs scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs.dim_size[1].
  INFER_ERROR("", op, "[4,2];[2,3];[6];[6]");
}

TEST(UniformQuantizedOpsTest,
     UniformQuantizedConvolutionShapeInferencePerTensor) {
  ShapeInferenceTestOp op("UniformQuantizedConvolution");
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolution")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_QINT8)
                   .Attr("Tout", DT_QINT32)
                   .Attr("lhs_quantization_min_val", kInt8Min)
                   .Attr("lhs_quantization_max_val", kInt8Max)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Attr("padding", "VALID")
                   .Finalize(&op.node_def));
  // Uses default Attrs (and default conv_params settings).
  // feature_group_count = 1
  // batch_group_count = 1
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]

  INFER_OK(op, "[2,3,40,50];[6,3,4,5];[];[];[];[];[];[]", "[2,6,37,46]");

  // lhs feature dimension size and rhs input feature dimension size must match.
  INFER_ERROR("", op, "[2,3,40,50];[6,9,4,5];[];[];[];[];[];[]");
  // lhs scales and zero_points must be scalar tensors.
  INFER_ERROR("", op, "[2,3,40,50];[6,3,4,5];[2];[2];[];[];[];[]");
  // scales and zero_points must have same rank.
  INFER_ERROR("scales and zero_points must have same rank.", op,
              "[2,3,40,50];[6,3,4,5];[];[];[6];[];[];[]");
  // Output scales and zero_points must be scalar tensors is rhs scales and
  // zero_points are scalar tensors.
  INFER_ERROR("", op, "[2,3,40,50];[6,3,4,5];[];[];[];[];[12];[12]");
}

TEST(UniformQuantizedOpsTest,
     UniformQuantizedConvolutionShapeInferencePerChannelRhs) {
  ShapeInferenceTestOp op("UniformQuantizedConvolution");
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolution")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_QINT8)
                   .Attr("Tout", DT_QINT32)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("lhs_quantization_min_val", kInt8Min)
                   .Attr("lhs_quantization_max_val", kInt8Max)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Attr("padding", "VALID")
                   .Finalize(&op.node_def));
  // Uses default Attrs (and default conv_params settings).
  // feature_group_count = 1
  // batch_group_count = 1
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]

  INFER_OK(op, "[2,3,40,50];[6,3,4,5];[];[];[6];[6];[];[]", "[2,6,37,46]");

  // If rhs scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs output feature dimension size.
  INFER_ERROR("", op, "[2,3,40,50];[6,3,4,5];[];[];[12];[12];[];[]");
}

TEST(UniformQuantizedOpsTest,
     UniformQuantizedConvolutionShapeInferencePerChannelRhsAndOutput) {
  ShapeInferenceTestOp op("UniformQuantizedConvolution");
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolution")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_QINT8)
                   .Attr("Tout", DT_QINT32)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("output_quantization_axis", 1)
                   .Attr("lhs_quantization_min_val", kInt8Min)
                   .Attr("lhs_quantization_max_val", kInt8Max)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Attr("padding", "VALID")
                   .Finalize(&op.node_def));
  // Uses default Attrs (and default conv_params settings).
  // feature_group_count = 1
  // batch_group_count = 1
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]

  INFER_OK(op, "[2,3,40,50];[6,3,4,5];[];[];[6];[6];[6];[6]", "[2,6,37,46]");
}

TEST(UniformQuantizedOpsTest,
     UniformQuantizedConvolutionHybridShapeInferencePerChannel) {
  ShapeInferenceTestOp op("UniformQuantizedConvolutionHybrid");
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_QINT8)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_QINT32)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Finalize(&op.node_def));
  // Uses default Attrs (and default conv_params settings).
  //
  // batch_group_count = 1
  // feature_group_count = 1
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]

  INFER_OK(op, "[2,3,40,50];[6,3,4,5];[6];[6]", "[2,6,37,46]");

  // If rhs scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs output feature dimension size.
  INFER_ERROR("", op, "[2,3,40,50];[6,3,4,5];[12];[12]");
}

}  // namespace tensorflow
