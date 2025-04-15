/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/ml_adjacent/algo/crop.h"

#include <cstring>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

using ::ml_adj::algo::Algo;
using ::ml_adj::data::OwningVectorRef;
using ::testing::ElementsAreArray;

namespace ml_adj {
namespace crop {
namespace {

// Gets a flattened vector from the given 4d coordinates whose values
// range from [0, prod(d1, d2, d3, d4)).
std::vector<float> GetIotaVec(dim_t d1, dim_t d2, dim_t d3, dim_t d4) {
  std::vector<float> res;
  res.resize(d1 * d2 * d3 * d4);
  std::iota(res.begin(), res.end(), 0);
  return res;
}

//----------------------------------------------------------------------------//
//                            CENTRAL CROP TESTS                              //
//----------------------------------------------------------------------------//

struct CropCenterTestParams {
  std::vector<dim_t> img_dims;
  std::vector<float> img_data;
  double frac;
  std::vector<float> expected_data;
  std::vector<dim_t> expected_shape;
};

class CropCenterTest : public testing::TestWithParam<CropCenterTestParams> {};

TEST_P(CropCenterTest, FloatPixelType) {
  const CropCenterTestParams& params = GetParam();

  // Image input.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // Frac input.
  OwningVectorRef frac(etype_t::f64);
  frac.Resize({1});
  ASSERT_EQ(frac.Bytes(), sizeof(double));
  std::memcpy(frac.Data(), &params.frac, frac.Bytes());

  // Empty output.
  OwningVectorRef output(etype_t::f32);

  const Algo* center_crop = Impl_CenterCrop();
  center_crop->process({&img, &frac}, {&output});

  ASSERT_EQ(output.Bytes(), params.expected_data.size() * sizeof(float));
  ASSERT_EQ(output.Dims(), params.expected_shape);

  const float* out_data = reinterpret_cast<float*>(output.Data());
  EXPECT_THAT(absl::MakeSpan(out_data, output.NumElements()),
              ElementsAreArray(params.expected_data));
}


INSTANTIATE_TEST_SUITE_P(
    CropTests, CropCenterTest,
    testing::ValuesIn({
        CropCenterTestParams{{1, 4, 4, 1},
                             GetIotaVec(1, 4, 4, 1),
                             0.5,
                             {5, 6, 9, 10},
                             {1, 2, 2, 1}},
        CropCenterTestParams{{1, 5, 5, 1},
                             GetIotaVec(1, 5, 5, 1),
                             0.5,
                             {6, 7, 8, 11, 12, 13, 16, 17, 18},
                             {1, 3, 3, 1}},
        CropCenterTestParams{{1, 3, 3, 1},
                             GetIotaVec(1, 3, 3, 1),
                             0.5,
                             {0, 1, 2, 3, 4, 5, 6, 7, 8},
                             {1, 3, 3, 1}},
        CropCenterTestParams{{1, 5, 5, 1},
                             GetIotaVec(1, 5, 5, 1),
                             0.9,
                             GetIotaVec(1, 5, 5, 1),
                             {1, 5, 5, 1}},
        CropCenterTestParams{
            {1, 5, 5, 1}, GetIotaVec(1, 5, 5, 1), 0.2, {12}, {1, 1, 1, 1}},
        CropCenterTestParams{{1, 2, 2, 2},
                             GetIotaVec(1, 2, 2, 2),
                             .7,
                             {0, 1, 2, 3, 4, 5, 6, 7},
                             {1, 2, 2, 2}},
        CropCenterTestParams{
            {1, 3, 3, 2}, GetIotaVec(1, 3, 3, 2), .1, {8, 9}, {1, 1, 1, 2}},
        CropCenterTestParams{
            {2, 3, 3, 1}, GetIotaVec(2, 3, 3, 1), .1, {4, 13}, {2, 1, 1, 1}},
        CropCenterTestParams{{2, 3, 3, 2},
                             GetIotaVec(2, 3, 3, 2),
                             .1,
                             {8, 9, 26, 27},
                             {2, 1, 1, 2}},
    }));

//----------------------------------------------------------------------------//
//                        CROP TO BOUNDING BOX TESTS                          //
//----------------------------------------------------------------------------//

struct CropToBoundingBoxTestParams {
  const std::vector<dim_t> img_dims;
  const std::vector<float> img_data;
  dim_t offset_height;
  dim_t offset_width;
  dim_t target_height;
  dim_t target_width;
  const std::vector<dim_t> expected_shape;
  const std::vector<float> expected_data;
};

class CropToBoundingBoxTest
    : public testing::TestWithParam<CropToBoundingBoxTestParams> {};

TEST_P(CropToBoundingBoxTest, FloatPixelType) {
  const CropToBoundingBoxTestParams& params = GetParam();
  // Image input.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // Offset height input.
  OwningVectorRef offset_height(etype_t::i32);
  offset_height.Resize({1});
  ASSERT_EQ(offset_height.Bytes(), sizeof(int));
  std::memcpy(offset_height.Data(), &params.offset_height,
              offset_height.Bytes());

  // Offset width input.
  OwningVectorRef offset_width(etype_t::i32);
  offset_width.Resize({1});
  ASSERT_EQ(offset_width.Bytes(), sizeof(int));
  std::memcpy(offset_width.Data(), &params.offset_width, offset_width.Bytes());

  // Target height input.
  OwningVectorRef target_height(etype_t::i32);
  target_height.Resize({1});
  ASSERT_EQ(target_height.Bytes(), sizeof(int));
  std::memcpy(target_height.Data(), &params.target_height,
              target_height.Bytes());

  // Target width input.
  OwningVectorRef target_width(etype_t::i32);
  target_width.Resize({1});
  ASSERT_EQ(target_width.Bytes(), sizeof(int));
  std::memcpy(target_width.Data(), &params.target_width, target_width.Bytes());

  // Empty output.
  OwningVectorRef output(etype_t::f32);
  const Algo* crop_to_bounding_box = Impl_CropToBoundingBox();
  crop_to_bounding_box->process(
      {&img, &offset_height, &offset_width, &target_height, &target_width},
      {&output});
  ASSERT_EQ(output.Bytes(), params.expected_data.size() * sizeof(float));
  ASSERT_EQ(output.Dims(), params.expected_shape);

  const float* out_data = reinterpret_cast<float*>(output.Data());
  EXPECT_THAT(absl::MakeSpan(out_data, output.NumElements()),
              ElementsAreArray(params.expected_data));
}

INSTANTIATE_TEST_SUITE_P(
    CropTests, CropToBoundingBoxTest,
    testing::ValuesIn({
        // Top-left corner.
        CropToBoundingBoxTestParams{{1, 5, 5, 1},
                                    GetIotaVec(1, 5, 5, 1),
                                    0,
                                    0,
                                    2,
                                    2,
                                    {1, 2, 2, 1},
                                    {0, 1,  //
                                     5, 6}},
        // Bottom-right corner.
        CropToBoundingBoxTestParams{{1, 5, 5, 1},
                                    GetIotaVec(1, 5, 5, 1),
                                    3,
                                    3,
                                    2,
                                    2,
                                    {1, 2, 2, 1},
                                    {18, 19,  //
                                     23, 24}},
        // Top-right corner.
        CropToBoundingBoxTestParams{{1, 5, 5, 1},
                                    GetIotaVec(1, 5, 5, 1),
                                    0,
                                    3,
                                    2,
                                    2,
                                    {1, 2, 2, 1},
                                    {3, 4,  //
                                     8, 9}},
        // Non-corner crop.
        CropToBoundingBoxTestParams{{1, 5, 5, 1},
                                    GetIotaVec(1, 5, 5, 1),
                                    2,
                                    1,
                                    3,
                                    3,
                                    {1, 3, 3, 1},
                                    {11, 12, 13,  //
                                     16, 17, 18,  //
                                     21, 22, 23}},
        // Full image size crop.
        CropToBoundingBoxTestParams{{1, 3, 3, 1},
                                    GetIotaVec(1, 3, 3, 1),
                                    0,
                                    0,
                                    3,
                                    3,
                                    {1, 3, 3, 1},
                                    {0, 1, 2,  //
                                     3, 4, 5,  //
                                     6, 7, 8}},
        // One-element size crop.
        CropToBoundingBoxTestParams{{1, 3, 3, 1},
                                    GetIotaVec(1, 3, 3, 1),
                                    1,
                                    1,
                                    1,
                                    1,
                                    {1, 1, 1, 1},
                                    {4}},
        // Multichannel image.
        CropToBoundingBoxTestParams{{1, 5, 5, 3},
                                    GetIotaVec(1, 5, 5, 3),
                                    2,
                                    2,
                                    2,
                                    2,
                                    {1, 2, 2, 3},
                                    {36, 37, 38, 39, 40, 41,  //
                                     51, 52, 53, 54, 55, 56}},
        // Multibatch multichannel image.
        CropToBoundingBoxTestParams{{2, 5, 5, 2},
                                    GetIotaVec(2, 5, 5, 2),
                                    2,
                                    2,
                                    2,
                                    2,
                                    {2, 2, 2, 2},
                                    {24, 25, 26, 27, 34, 35, 36, 37,  //
                                     74, 75, 76, 77, 84, 85, 86, 87}},
    }));

}  // namespace
}  // namespace crop
}  // namespace ml_adj
