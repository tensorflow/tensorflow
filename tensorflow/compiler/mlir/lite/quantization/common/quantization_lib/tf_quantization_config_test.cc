/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/tf_quantization_config.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace TFL {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

TEST(TfQuantizationConfigTest, ConvertDataTypeToInferenceDataType) {
  const std::vector<std::pair<tensorflow::DataType, InferenceDataType>> cases =
      {
          {tensorflow::DT_INVALID, InferenceDataType::kInvalid},
          {tensorflow::DT_FLOAT, InferenceDataType::kFloat},
          {tensorflow::DT_HALF, InferenceDataType::kHalf},
          {tensorflow::DT_INT8, InferenceDataType::kInt8},
          {tensorflow::DT_UINT8, InferenceDataType::kUInt8},
          {tensorflow::DT_INT16, InferenceDataType::kInt16},
          {tensorflow::DT_UINT16, InferenceDataType::kUInt16},
          {tensorflow::DT_INT32, InferenceDataType::kInt32},
          {tensorflow::DT_QINT8, InferenceDataType::kQInt8},
          {tensorflow::DT_QUINT8, InferenceDataType::kQUInt8},
          {tensorflow::DT_QINT16, InferenceDataType::kQInt16},
          {tensorflow::DT_QUINT16, InferenceDataType::kQUInt16},
          {tensorflow::DT_QINT32, InferenceDataType::kQInt32},
          {tensorflow::DT_BOOL, InferenceDataType::kInvalid},
      };
  for (const auto& [data_type, expected] : cases) {
    const InferenceDataType actual =
        ConvertDataTypeToInferenceDataType(data_type);
    EXPECT_EQ(actual, expected) << tensorflow::DataType_Name(data_type);
    if (expected != InferenceDataType::kInvalid) {
      EXPECT_EQ(InferenceDataTypeName(actual),
                tensorflow::DataType_Name(data_type));
    }
  }
}

TEST(TfQuantizationConfigTest, ParseInputNodeQuantSpecsQuantized) {
  QuantizationSpecs quant_specs;
  ASSERT_FALSE(ParseInputNodeQuantSpecs("a,b", "-1,0", "1,255", "DT_QUINT8",
                                        &quant_specs));
  EXPECT_EQ(quant_specs.inference_type, InferenceDataType::kQUInt8);
  EXPECT_THAT(
      quant_specs.input_ranges,
      ElementsAre(
          Pair(std::optional<double>(-1.0), std::optional<double>(1.0)),
          Pair(std::optional<double>(0.0), std::optional<double>(255.0))));
}

TEST(TfQuantizationConfigTest, ParseInputNodeQuantSpecsFloatIgnoresRanges) {
  QuantizationSpecs quant_specs;
  ASSERT_FALSE(ParseInputNodeQuantSpecs("a", "-1", "1", "", &quant_specs));
  EXPECT_EQ(quant_specs.inference_type, InferenceDataType::kFloat);
  EXPECT_TRUE(quant_specs.input_ranges.empty());
}

TEST(TfQuantizationConfigTest, GetInputNodeQuantSpecsUnsupportedDataType) {
  QuantizationSpecs quant_specs;
  ASSERT_FALSE(GetInputNodeQuantSpecs({"a"}, {-1.0}, {1.0},
                                      tensorflow::DT_DOUBLE, &quant_specs));
  EXPECT_EQ(quant_specs.inference_type, InferenceDataType::kInvalid);
  EXPECT_TRUE(quant_specs.input_ranges.empty());
}

TEST(TfQuantizationConfigTest, GetInputNodeQuantSpecsRangeCountMismatch) {
  QuantizationSpecs quant_specs;
  EXPECT_TRUE(GetInputNodeQuantSpecs({"a", "b"}, {-1.0}, {1.0},
                                     tensorflow::DT_QUINT8, &quant_specs));
}

TEST(TfQuantizationConfigTest, ParseInputNodeQuantSpecsInvalidType) {
  QuantizationSpecs quant_specs;
  EXPECT_TRUE(
      ParseInputNodeQuantSpecs("a", "-1", "1", "NOT_A_TYPE", &quant_specs));
}

}  // namespace
}  // namespace TFL
}  // namespace mlir
