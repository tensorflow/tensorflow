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

// Gets a flattened vector from the given 4d coordinates whose values
// range from [0, prod(d1, d2, d3, d4)).
std::vector<float> GetIotaVec(dim_t d1, dim_t d2, dim_t d3, dim_t d4) {
  std::vector<float> res;
  res.resize(d1 * d2 * d3 * d4);
  std::iota(res.begin(), res.end(), 0);
  return res;
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

}  // namespace
}  // namespace crop
}  // namespace ml_adj
