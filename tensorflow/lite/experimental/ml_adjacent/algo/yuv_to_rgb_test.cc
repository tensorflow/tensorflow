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
#include "tensorflow/lite/experimental/ml_adjacent/algo/yuv_to_rgb.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

using ::ml_adj::algo::Algo;
using ::ml_adj::data::OwningVectorRef;

namespace ml_adj {
namespace yuv_to_rgb {
namespace {

struct YuvToRgbTestParams {
  const std::vector<dim_t> img_dims;
  const std::vector<float> img_data;
  const std::vector<float> expected_data;
  const std::vector<dim_t> expected_shape;
};

class YuvToRgbTest : public ::testing::TestWithParam<YuvToRgbTestParams> {};

TEST_P(YuvToRgbTest, FloatPixelType) {
  constexpr float kAbsError = 0.1f;
  const YuvToRgbTestParams& params = GetParam();

  // Image input.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // Empty output image.
  OwningVectorRef output(etype_t::f32);

  // Convert YUV to RGB image.
  const Algo* yuv_to_rgb = Impl_YuvToRgb();
  yuv_to_rgb->process({&img}, {&output});

  // Check resize output.
  ASSERT_EQ(output.Bytes(), params.expected_data.size() * sizeof(float));
  ASSERT_EQ(output.Dims(), params.expected_shape);

  const float* out_data = reinterpret_cast<float*>(output.Data());
  for (int i = 0; i < output.NumElements(); ++i) {
    EXPECT_NEAR(out_data[i], params.expected_data[i], kAbsError)
        << "out_data[" << i << "] = " << out_data[i] << ", expected_data[" << i
        << "] = " << params.expected_data[i];
  }
}

INSTANTIATE_TEST_SUITE_P(
    YuvToRgbTests, YuvToRgbTest,
    testing::ValuesIn({
        YuvToRgbTestParams{/*img_dims=*/{1, 3, 2, 3},
                           /*img_data=*/
                           {
                               92.5f,
                               58.3f,
                               -71.5f,
                               93.5f,
                               58.3f,
                               -71.5f,  //
                               102.5f,
                               58.3f,
                               -71.5f,
                               103.5f,
                               58.3f,
                               -71.5f,  //
                               112.5f,
                               58.3f,
                               -71.5f,
                               113.5f,
                               58.3f,
                               -71.5f,
                           },
                           /*expected_data=*/
                           {11, 111, 211, 12, 112, 212,  //
                            21, 121, 221, 22, 122, 222,  //
                            31, 131, 231, 32, 132, 232},
                           /*expected_shape=*/{1, 3, 2, 3}},
        YuvToRgbTestParams{/*img_dims=*/{2, 3, 2, 3},
                           /*img_data=*/
                           {92.5f,  58.3f, -71.5f, 93.5f,  58.3f, -71.5f,  //
                            102.5f, 58.3f, -71.5f, 103.5f, 58.3f, -71.5f,  //
                            112.5f, 58.3f, -71.5f, 113.5f, 58.3f, -71.5f,  //
                            92.5f,  58.3f, -71.5f, 93.5f,  58.3f, -71.5f,  //
                            102.5f, 58.3f, -71.5f, 103.5f, 58.3f, -71.5f,  //
                            112.5f, 58.3f, -71.5f, 113.5f, 58.3f, -71.5f},
                           /*expected_data=*/
                           {11, 111, 211, 12, 112, 212,  //
                            21, 121, 221, 22, 122, 222,  //
                            31, 131, 231, 32, 132, 232,  //
                            11, 111, 211, 12, 112, 212,  //
                            21, 121, 221, 22, 122, 222,  //
                            31, 131, 231, 32, 132, 232},
                           /*expected_shape=*/{2, 3, 2, 3}},
    }));

}  // namespace
}  // namespace yuv_to_rgb
}  // namespace ml_adj
