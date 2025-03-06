// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

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
}  // namespace
}  // namespace qnn
