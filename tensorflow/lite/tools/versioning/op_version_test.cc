/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/op_version.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {

TEST(OpVersionTest, VersioningSpareToDense) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT8,
                                             TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .input_types = std::vector<TensorType>{TensorType_UINT8, TensorType_UINT8,
                                             TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .input_types = std::vector<TensorType>{TensorType_INT64, TensorType_INT64,
                                             TensorType_INT64},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .input_types = std::vector<TensorType>{TensorType_INT32, TensorType_INT32,
                                             TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

// Test version for a simple Op with 2 versions and the input type controls the
// version.
void SimpleVersioningTest(BuiltinOperator op) {
  OpSignature fake_op_sig = {
      .op = op,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = op,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

// Test version for a simple Op with 2 versions and the output type controls the
void SimpleOutputVersioningTest(BuiltinOperator op) {
  OpSignature fake_op_sig = {
      .op = op,
      .input_types = std::vector<TensorType>{},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = op,
      .input_types = std::vector<TensorType>{},
      .output_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningEqualTest) {
  SimpleVersioningTest(BuiltinOperator_EQUAL);
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_EQUAL,
      .input_types = std::vector<TensorType>{TensorType_STRING},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningNotEqualTest) {
  SimpleVersioningTest(BuiltinOperator_NOT_EQUAL);
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_NOT_EQUAL,
      .input_types = std::vector<TensorType>{TensorType_STRING},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningLessTest) {
  SimpleVersioningTest(BuiltinOperator_LESS);
}

TEST(OpVersionTest, VersioningLessEqualTest) {
  SimpleVersioningTest(BuiltinOperator_LESS_EQUAL);
}

TEST(OpVersionTest, VersioningGreaterTest) {
  SimpleVersioningTest(BuiltinOperator_GREATER);
}

TEST(OpVersionTest, VersioningGreaterEqualTest) {
  SimpleVersioningTest(BuiltinOperator_GREATER_EQUAL);
}

TEST(OpVersionTest, VersioningSpaceToBatchNDTest) {
  SimpleVersioningTest(BuiltinOperator_NOT_EQUAL);
}

TEST(OpVersionTest, VersioningLogSoftmaxTest) {
  SimpleVersioningTest(BuiltinOperator_LOG_SOFTMAX);
}

TEST(OpVersionTest, VersioningPackTest) {
  SimpleVersioningTest(BuiltinOperator_PACK);
}

TEST(OpVersionTest, VersioningUnpackTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .input_types = std::vector<TensorType>{TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningReluTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .input_types = std::vector<TensorType>{TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningBatchToSpaceNDTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BATCH_TO_SPACE_ND,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  fake_op_sig.options.single_input_op.num_dims = 3;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.options.single_input_op.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_BATCH_TO_SPACE_ND,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  fake_op_sig.options.single_input_op.num_dims = 3;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.options.single_input_op.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningTanhTest) {
  SimpleVersioningTest(BuiltinOperator_TANH);
}

TEST(OpVersionTest, VersioningStridedSliceTest) {
  SimpleVersioningTest(BuiltinOperator_STRIDED_SLICE);
}

TEST(OpVersionTest, VersioningSpaceToDepthTest) {
  SimpleVersioningTest(BuiltinOperator_SPACE_TO_DEPTH);
}

TEST(OpVersionTest, VersioningSliceTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .input_types = std::vector<TensorType>{TensorType_STRING},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningLogisticTest) {
  SimpleVersioningTest(BuiltinOperator_SPACE_TO_DEPTH);
}

TEST(OpVersionTest, VersioningL2NormTest) {
  SimpleOutputVersioningTest(BuiltinOperator_L2_NORMALIZATION);
}

TEST(OpVersionTest, VersioningMaxTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MAXIMUM,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };

  fake_op_sig.options.broadcast.need_broadcast = true;
  fake_op_sig.options.broadcast.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.options.broadcast.need_broadcast = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.options.broadcast.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_MAXIMUM,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  fake_op_sig.options.broadcast.need_broadcast = true;
  fake_op_sig.options.broadcast.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.options.broadcast.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningMinTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MINIMUM,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };

  fake_op_sig.options.broadcast.need_broadcast = true;
  fake_op_sig.options.broadcast.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.options.broadcast.need_broadcast = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.options.broadcast.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_MINIMUM,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  fake_op_sig.options.broadcast.need_broadcast = true;
  fake_op_sig.options.broadcast.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.options.broadcast.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningMeanTest) {
  SimpleVersioningTest(BuiltinOperator_MEAN);
}

TEST(OpVersionTest, VersioningSumTest) {
  SimpleVersioningTest(BuiltinOperator_SUM);
}

TEST(OpVersionTest, VersioningAddTest) {
  SimpleVersioningTest(BuiltinOperator_ADD);
}

TEST(OpVersionTest, VersioningSubTest) {
  SimpleVersioningTest(BuiltinOperator_SUB);
}

void SimpleMulVersioningTest(TensorType data_type, float multiplier,
                             int version) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MUL,
      .input_types = std::vector<TensorType>{data_type, data_type},
      .output_types = std::vector<TensorType>{data_type},
  };
  fake_op_sig.options.mul = {1.0f, 1.0f, 1.0f / multiplier};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), version);
}

TEST(OpVersionTest, VersioningMulTest) {
  SimpleMulVersioningTest(TensorType_UINT8, 0.5f, 1);
  SimpleMulVersioningTest(TensorType_INT8, 0.5f, 2);
  SimpleMulVersioningTest(TensorType_INT8, 2.0f, 3);
}

TEST(OpVersionTest, VersioningPadTest) {
  SimpleVersioningTest(BuiltinOperator_PAD);
}

TEST(OpVersionTest, VersioningPadV2Test) {
  SimpleVersioningTest(BuiltinOperator_PADV2);
}

TEST(OpVersionTest, VersioningConcatenationTest) {
  SimpleVersioningTest(BuiltinOperator_CONCATENATION);
}

TEST(OpVersionTest, VersioningSelectTest) {
  SimpleVersioningTest(BuiltinOperator_SELECT);
}

TEST(OpVersionTest, VersioningRelu6Test) {
  SimpleVersioningTest(BuiltinOperator_RELU6);
}

TEST(OpVersionTest, VersioningFullyConnectedTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .input_types =
          std::vector<TensorType>{TensorType_UINT8, TensorType_UINT8},
      .output_types = std::vector<TensorType>{TensorType_UINT8},
  };
  fake_op_sig.options.fully_connected = {
      false, FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT8},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  fake_op_sig.options.fully_connected = {
      false, FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);
}

TEST(OpVersionTest, VersioningDequantizeTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .input_types = std::vector<TensorType>{TensorType_INT16},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .input_types = std::vector<TensorType>{TensorType_FLOAT16},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .input_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningConv2DTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .input_types =
          std::vector<TensorType>{TensorType_UINT8, TensorType_UINT8},
      .output_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT8},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_INT8},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningFloorDivOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .input_types = std::vector<TensorType>{TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .input_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningTransposeConvOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .input_types = std::vector<TensorType>{TensorType_INT32, TensorType_INT8,
                                             TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .input_types = std::vector<TensorType>{TensorType_INT32, TensorType_INT8,
                                             TensorType_INT8, TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  const auto none_type = static_cast<::tflite::TensorType>(-1);
  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .input_types = std::vector<TensorType>{TensorType_INT32, TensorType_INT8,
                                             TensorType_INT8, none_type},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningSVDFOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_FLOAT32,
                                  TensorType_FLOAT32, TensorType_FLOAT32,
                                  TensorType_FLOAT32},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_INT8,
                                  TensorType_FLOAT32, TensorType_FLOAT32,
                                  TensorType_FLOAT32},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT8,
                                             TensorType_INT16, TensorType_INT32,
                                             TensorType_INT16},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}
TEST(OpVersionTest, VersioningDepthwiseConv2DTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_INT8},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT8},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_FLOAT32},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  fake_op_sig.options.depthwise_conv_2d.dilation_w_factor = 2;
  fake_op_sig.options.depthwise_conv_2d.dilation_h_factor = 2;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_FLOAT32},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  fake_op_sig.options.depthwise_conv_2d.dilation_w_factor = 1;
  fake_op_sig.options.depthwise_conv_2d.dilation_h_factor = 1;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningTileOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TILE,
      .input_types = std::vector<TensorType>{TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_TILE,
      .input_types = std::vector<TensorType>{TensorType_STRING},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningTransposeTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .input_types = std::vector<TensorType>{TensorType_BOOL},
  };
  fake_op_sig.options.single_input_op.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.options.single_input_op.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .input_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningGatherNdOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .input_types =
          std::vector<TensorType>{TensorType_INT32, TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .input_types =
          std::vector<TensorType>{TensorType_STRING, TensorType_INT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningDivTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DIV,
      .input_types = std::vector<TensorType>{TensorType_UINT8},
  };

  fake_op_sig.options.broadcast.need_broadcast = true;
  fake_op_sig.options.broadcast.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.options.broadcast.need_broadcast = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
  fake_op_sig.options.broadcast.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTEst, VersioningFillTest) {
  OpSignature fake_op_sig = {.op = BuiltinOperator_FILL,
                             .input_types = std::vector<TensorType>{
                                 TensorType_INT32, TensorType_BOOL}};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig = {.op = BuiltinOperator_FILL,
                 .input_types = std::vector<TensorType>{TensorType_INT32,
                                                        TensorType_STRING}};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig = {.op = BuiltinOperator_FILL,
                 .input_types = std::vector<TensorType>{TensorType_INT32,
                                                        TensorType_INT32}};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningResizeBilinearTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_INT32},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // align_corners=true is still version 1.
  fake_op_sig.options.resize.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // half_pixel_centers=true must be version 3.
  fake_op_sig.options.resize.align_corners = false;
  fake_op_sig.options.resize.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT32},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.options.resize.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}
TEST(OpVersionTest, VersioningResizeNearestNeighborTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .input_types =
          std::vector<TensorType>{TensorType_FLOAT32, TensorType_INT32},
      .output_types = std::vector<TensorType>{TensorType_FLOAT32},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // align_corners=true is version 3.
  fake_op_sig.options.resize.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // half_pixel_centers=true must be version 3.
  fake_op_sig.options.resize.align_corners = false;
  fake_op_sig.options.resize.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .input_types = std::vector<TensorType>{TensorType_INT8, TensorType_INT32},
      .output_types = std::vector<TensorType>{TensorType_INT8},
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.options.resize.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}
}  // namespace tflite
