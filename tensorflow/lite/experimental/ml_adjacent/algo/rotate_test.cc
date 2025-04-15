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
#include "tensorflow/lite/experimental/ml_adjacent/algo/rotate.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

using ::ml_adj::algo::Algo;
using ::ml_adj::data::OwningVectorRef;

namespace ml_adj {
namespace rotate {
namespace {

struct RotateTestParams {
  const std::vector<dim_t> img_dims;
  const std::vector<float> img_data;
  const int angle;
  const std::vector<float> expected_data;
  const std::vector<dim_t> expected_shape;
};

class RotateTest : public ::testing::TestWithParam<RotateTestParams> {};

TEST_P(RotateTest, FloatPixelType) {
  constexpr float kAbsError = 0.01f;

  const RotateTestParams& params = GetParam();

  // Image input.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // Specify angle for rotation.
  OwningVectorRef angle(etype_t::i32);
  angle.Resize({1});
  ASSERT_EQ(angle.Bytes(), sizeof(int));
  std::memcpy(angle.Data(), &params.angle, angle.Bytes());

  // Empty output image.
  OwningVectorRef output(etype_t::f32);

  // Run image rotate custom call.
  const Algo* rotate = Impl_Rotate();
  rotate->process({&img, &angle}, {&output});

  // Check rotate output.
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
    RotateTests, RotateTest,
    testing::ValuesIn({
        // 4-D Tensor of shape [batch, height, width, channels] is used below.

        RotateTestParams{/*img_dims=*/{1, 3, 3, 1},
                         /*img_data=*/
                         {11, 12, 13,  //
                          21, 22, 23,  //
                          31, 32, 33},
                         /*angle=*/90,
                         /*expected_data=*/
                         {31, 21, 11,  //
                          32, 22, 12,  //
                          33, 23, 13},
                         /*expected_shape=*/{1, 3, 3, 1}},

        RotateTestParams{/*img_dims=*/{1, 3, 3, 1},
                         /*img_data=*/
                         {11, 12, 13,  //
                          21, 22, 23,  //
                          31, 32, 33},
                         /*angle=*/180,
                         /*expected_data=*/
                         {33, 32, 31,  //
                          23, 22, 21,  //
                          13, 12, 11},
                         /*expected_shape=*/{1, 3, 3, 1}},

        RotateTestParams{/*img_dims=*/{1, 3, 3, 1},
                         /*img_data=*/
                         {11, 12, 13,  //
                          21, 22, 23,  //
                          31, 32, 33},
                         /*angle=*/270,
                         /*expected_data=*/
                         {13, 23, 33,  //
                          12, 22, 32,  //
                          11, 21, 31},
                         /*expected_shape=*/{1, 3, 3, 1}},

        RotateTestParams{/*img_dims=*/{1, 8, 8, 1},
                         /*img_data=*/
                         {1, 1, 1, 1, 1, 1, 1, 1,  //
                          1, 0, 0, 0, 0, 0, 0, 1,  //
                          1, 0, 0, 0, 0, 0, 0, 1,  //
                          1, 0, 0, 0, 0, 0, 0, 1,  //
                          1, 0, 0, 0, 0, 0, 0, 1,  //
                          1, 0, 0, 0, 0, 0, 0, 1,  //
                          1, 0, 0, 0, 0, 0, 0, 1,  //
                          1, 1, 1, 1, 1, 1, 1, 1},
                         /*angle=*/-45,
                         /*expected_data=*/
                         {
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.59f,
                             0.83f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.00f, 0.54f, 0.00f,
                             0.12f, 0.83f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.54f, 0.00f, 0.00f,
                             0.00f, 0.12f, 0.83f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.54f, 0.00f, 0.00f, 0.00f,
                             0.00f, 0.00f, 0.12f, 0.83f, 0.00f, 0.00f,  //
                             0.00f, 0.78f, 0.00f, 0.00f, 0.00f, 0.00f,
                             0.00f, 0.00f, 0.00f, 0.23f, 0.97f, 0.00f,  //
                             0.00f, 0.00f, 0.54f, 0.00f, 0.00f, 0.00f,
                             0.00f, 0.00f, 0.12f, 0.83f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.54f, 0.00f, 0.00f,
                             0.00f, 0.12f, 0.83f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.00f, 0.54f, 0.00f,
                             0.12f, 0.83f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.59f,
                             0.83f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
                             0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,  //
                         },
                         /*expected_shape=*/{1, 12, 12, 1}},
    }));

}  // namespace
}  // namespace rotate
}  // namespace ml_adj
