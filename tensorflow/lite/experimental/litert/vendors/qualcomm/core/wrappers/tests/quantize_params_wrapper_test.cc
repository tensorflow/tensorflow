// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

#include <gtest/gtest.h>

namespace qnn {
namespace {

TEST(UndefinedQuantizeParamsWrapperTest, DefaultConstructorTest) {
  UndefinedQuantizeParamsWrapper wrapper;
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(UndefinedQuantizeParamsWrapperTest, CopyConstructorTest) {
  UndefinedQuantizeParamsWrapper wrapper1;
  UndefinedQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(UndefinedQuantizeParamsWrapperTest, MoveConstructorTest) {
  UndefinedQuantizeParamsWrapper wrapper1;
  UndefinedQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, ConstructorTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper(scale, zero_point);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(dst.scaleOffsetEncoding.scale, scale);
  EXPECT_EQ(dst.scaleOffsetEncoding.offset, -zero_point);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, CopyConstructorTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper1(scale, zero_point);
  ScaleOffsetQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(dst.scaleOffsetEncoding.scale, scale);
  EXPECT_EQ(dst.scaleOffsetEncoding.offset, -zero_point);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, MoveConstructorTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper1(scale, zero_point);
  ScaleOffsetQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(dst.scaleOffsetEncoding.scale, scale);
  EXPECT_EQ(dst.scaleOffsetEncoding.offset, -zero_point);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, QnnConstructorTest) {
  ScaleOffsetQuantizeParamsWrapper wrapper1(1.5f, 10);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  ScaleOffsetQuantizeParamsWrapper wrapper2(dst1.scaleOffsetEncoding);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst2);
  EXPECT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  EXPECT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  EXPECT_FLOAT_EQ(dst1.scaleOffsetEncoding.scale,
                  dst2.scaleOffsetEncoding.scale);
  EXPECT_EQ(dst1.scaleOffsetEncoding.offset, dst2.scaleOffsetEncoding.offset);
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, ConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper(axis, scales, zero_points);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.axis, axis);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.numScaleOffsets, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].scale,
                    scales[i]);
    EXPECT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].offset,
              -zero_points[i]);
  }
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, CopyConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.axis, axis);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.numScaleOffsets, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].scale,
                    scales[i]);
    EXPECT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].offset,
              -zero_points[i]);
  }
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, MoveConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.axis, axis);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.numScaleOffsets, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].scale,
                    scales[i]);
    EXPECT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].offset,
              -zero_points[i]);
  }
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, QnnConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(dst1.axisScaleOffsetEncoding);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst2);
  EXPECT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  EXPECT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  EXPECT_EQ(dst1.axisScaleOffsetEncoding.numScaleOffsets,
            dst2.axisScaleOffsetEncoding.numScaleOffsets);
  for (size_t i = 0; i < dst1.axisScaleOffsetEncoding.numScaleOffsets; ++i) {
    EXPECT_EQ(dst1.axisScaleOffsetEncoding.scaleOffset[i].scale,
              dst2.axisScaleOffsetEncoding.scaleOffset[i].scale);
    EXPECT_EQ(dst1.axisScaleOffsetEncoding.scaleOffset[i].offset,
              dst2.axisScaleOffsetEncoding.scaleOffset[i].offset);
  }
}
}  // namespace
}  // namespace qnn
