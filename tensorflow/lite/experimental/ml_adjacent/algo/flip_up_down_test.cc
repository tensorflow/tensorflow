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
#include "tensorflow/lite/experimental/ml_adjacent/algo/flip_up_down.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

using ::ml_adj::algo::Algo;
using ::ml_adj::data::OwningVectorRef;

namespace ml_adj {
namespace flip_up_down {
namespace {

struct FlipUpDownTestParams {
  const std::vector<dim_t> img_dims;
  const std::vector<float> img_data;
  const std::vector<float> expected_data;
  const std::vector<dim_t> expected_shape;
};

class FlipUpDownTest : public ::testing::TestWithParam<FlipUpDownTestParams> {};

TEST_P(FlipUpDownTest, FloatPixelType) {
  constexpr float kAbsError = 0.01f;
  const FlipUpDownTestParams& params = GetParam();

  // Image input.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // Empty output image.
  OwningVectorRef output(etype_t::f32);

  // Flip image vertically.
  const Algo* flip_up_down = Impl_FlipUpDown();
  flip_up_down->process({&img}, {&output});

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
    FlipUpDownTests, FlipUpDownTest,
    testing::ValuesIn({
        FlipUpDownTestParams{/*img_dims=*/{1, 3, 3, 1},
                             /*img_data=*/
                             {11, 12, 13,  //
                              21, 22, 23,  //
                              31, 32, 33},
                             /*expected_data=*/
                             {31, 32, 33,  //
                              21, 22, 23,  //
                              11, 12, 13},
                             /*expected_shape=*/{1, 3, 3, 1}},
        FlipUpDownTestParams{/*img_dims=*/{1, 3, 3, 2},
                             /*img_data=*/
                             {11, 2, 12, 3, 13, 4,  //
                              21, 3, 22, 4, 23, 5,  //
                              31, 4, 32, 5, 33, 6},
                             /*expected_data=*/
                             {31, 4, 32, 5, 33, 6,  //
                              21, 3, 22, 4, 23, 5,  //
                              11, 2, 12, 3, 13, 4},
                             /*expected_shape=*/{1, 3, 3, 2}},
        FlipUpDownTestParams{/*img_dims=*/{2, 3, 3, 2},
                             /*img_data=*/
                             {11, 2, 12, 3, 13, 4,  //
                              21, 3, 22, 4, 23, 5,  //
                              31, 4, 32, 5, 33, 6,  //
                              //
                              13, 4, 12, 3, 11, 2,  //
                              23, 5, 22, 4, 21, 3,  //
                              33, 6, 32, 5, 31, 4},
                             /*expected_data=*/
                             {31, 4, 32, 5, 33, 6,  //
                              21, 3, 22, 4, 23, 5,  //
                              11, 2, 12, 3, 13, 4,  //
                              //
                              33, 6, 32, 5, 31, 4,  //
                              23, 5, 22, 4, 21, 3,  //
                              13, 4, 12, 3, 11, 2},
                             /*expected_shape=*/{2, 3, 3, 2}},
    }));

}  // namespace
}  // namespace flip_up_down
}  // namespace ml_adj
