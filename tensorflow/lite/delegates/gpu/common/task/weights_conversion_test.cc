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

#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

TEST(WeightsConversionTest, GetTotalElementsCountForLayoutOverflow_k2DX4I4) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  OHWI shape(1, 1, 1, 2000000000);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 0);
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayoutOverflow_kOICustom) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::kOICustomSpatialI4O4;
  weight_desc.spatial_remap.resize(1);

  OHWI shape(1, 1, 1, 2000000000);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 0);
}

TEST(WeightsConversionTest,
     GetTotalElementsCountForLayoutOverflow_OHWDI_k2DX4I4) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  OHWDI shape(1, 1, 1, 1, 2000000000);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 0);
}

TEST(WeightsConversionTest,
     GetTotalElementsCountForLayoutOverflow_OHWDI_kOICustom) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::kOICustomSpatialI4O4;
  weight_desc.spatial_remap.resize(1);

  OHWDI shape(1, 1, 1, 1, 2000000000);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 0);
}

TEST(WeightsConversionTest, RearrangeWeightsEarlyReturnOnOverflow_k2DX4I4) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 1, 1, 2000000000);

  std::vector<uint8_t> dummy_output(4, 0xAB);
  RearrangeWeights(weights, weight_desc, absl::MakeSpan(dummy_output));
  EXPECT_EQ(dummy_output[0], 0xAB);
  EXPECT_EQ(dummy_output[1], 0xAB);
  EXPECT_EQ(dummy_output[2], 0xAB);
  EXPECT_EQ(dummy_output[3], 0xAB);
}

TEST(WeightsConversionTest, RearrangeWeightsEarlyReturnOnOverflow_OHWDI) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  Tensor<OHWDI, DataType::FLOAT32> weights;
  weights.shape =
      OHWDI(1, 1, 1, 1, 2000000000);  // O, H, W, D, I? Or O, D, H, W, I?

  std::vector<uint8_t> dummy_output(4, 0xAB);
  RearrangeWeights(weights, weight_desc, absl::MakeSpan(dummy_output));
  EXPECT_EQ(dummy_output[0], 0xAB);
  EXPECT_EQ(dummy_output[1], 0xAB);
  EXPECT_EQ(dummy_output[2], 0xAB);
  EXPECT_EQ(dummy_output[3], 0xAB);
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayoutNormal_k2DX4I4) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  OHWI shape(2, 2, 2, 8);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 128);
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayoutNormal_kOICustom) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::kOICustomSpatialI4O4;
  weight_desc.spatial_remap.resize(3);

  OHWI shape(2, 2, 2, 8);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 96);
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayoutUnknown) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::kUnknown;

  OHWI shape(1, 1, 1, 1);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape),
            static_cast<uint>(-1));
}

TEST(WeightsConversionTest, RearrangeWeightsNormal_k2DX4I4) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 1, 1, 1);
  weights.data.resize(1);

  std::vector<uint8_t> dummy_output(64);
  RearrangeWeights(weights, weight_desc, absl::MakeSpan(dummy_output));
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayout_Valid_k2DX4I4) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  weight_desc.output_group_size = 1;

  OHWI shape(1, 1, 1, 1);
  // i_aligned = 4, o_aligned = 4, h=1, w=1, d=1
  // total = 4 * 4 * 1 * 1 * 1 = 16
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 16);
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayout_Valid_kOICustom) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::kOICustomSpatialI4O4;
  weight_desc.spatial_remap.resize(2);

  OHWI shape(1, 1, 1, 1);
  // i_aligned = 4, o_aligned = 4, remap_size = 2
  // total = 4 * 4 * 2 = 32
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape), 32);
}

TEST(WeightsConversionTest, GetTotalElementsCountForLayout_UnknownLayout) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.layout = WeightsLayout::kUnknown;
  OHWI shape(1, 1, 1, 1);
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, shape),
            static_cast<uint>(-1));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
