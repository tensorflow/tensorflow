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
#include "tensorflow/lite/experimental/ml_adjacent/algo/per_image_standardization.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

using ::ml_adj::algo::Algo;
using ::ml_adj::data::OwningVectorRef;

namespace ml_adj {
namespace per_image_standardization {
namespace {

struct PerImageStandardizationTestParams {
  const std::vector<dim_t> img_dims;
  const std::vector<float> img_data;
  const std::vector<float> expected_data;
};

class PerImageStandardizationTest
    : public testing::TestWithParam<PerImageStandardizationTestParams> {};

TEST_P(PerImageStandardizationTest, FloatPixelType) {
  const PerImageStandardizationTestParams& params = GetParam();

  // Create input image tensor.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // Create empty output image.
  OwningVectorRef output(etype_t::f32);

  // Run per image standardization custom call.
  const Algo* per_image_standardization = Impl_PerImageStandardization();
  per_image_standardization->process({&img}, {&output});

  // Check output.
  ASSERT_EQ(output.Bytes(), params.expected_data.size() * sizeof(float));
  ASSERT_EQ(output.Dims(), img.Dims());
  constexpr float kAbsError = 0.01f;
  float* out_data = reinterpret_cast<float*>(output.Data());
  for (int i = 0; i < output.NumElements(); ++i) {
    // TODO: b/298483909 - This print is being used in multiple locations.
    // Ideally should be moved to a shared test_utils file.
    EXPECT_NEAR(out_data[i], params.expected_data[i], kAbsError)
        << "out_data[" << i << "] = " << out_data[i] << ", expected_data[" << i
        << "] = " << params.expected_data[i];
  }
}

INSTANTIATE_TEST_SUITE_P(
    PerImageStandardizationTests, PerImageStandardizationTest,
    testing::ValuesIn({
        // 4-D Tensor of shape [batch, height, width, channels] is used below.
        PerImageStandardizationTestParams{/*img_dims=*/
                                          {1, 2, 2, 1},
                                          /*img_data=*/
                                          {1, 2,  //
                                           3, 4},
                                          /*expected_data=*/
                                          {-1.3416407, -0.4472136,  //
                                           0.4472136, 1.3416407}},
        // Two images in batch.
        PerImageStandardizationTestParams{/*img_dims=*/
                                          {2, 2, 2, 1},
                                          /*img_data=*/
                                          {1, 2,  //
                                           3, 4,  //
                                                  //
                                           1, 2,  //
                                           4, 8},
                                          /*expected_data=*/
                                          {-1.3416407, -0.4472136,   //
                                           0.4472136, 1.3416407,     //
                                                                     //
                                           -1.0257553, -0.65275335,  //
                                           0.09325048, 1.5852581}},
        // Multi-channel multi-batch image.
        PerImageStandardizationTestParams{/*img_dims=*/
                                          {2, 2, 2, 2},
                                          /*img_data=*/
                                          {1, 2,  //
                                           1, 3,  //
                                           1, 4,  //
                                           1, 5,  //
                                                  //
                                           1, 2,  //
                                           2, 2,  //
                                           3, 2,  //
                                           4, 2},
                                          /*expected_data=*/
                                          {-0.8451542, -0.16903085,   //
                                           -0.8451542, 0.50709254,    //
                                           -0.8451542, 1.1832159,     //
                                           -0.8451542, 1.8593392,     //
                                                                      //
                                           -1.5075567, -0.30151135,   //
                                           -0.30151135, -0.30151135,  //
                                           0.904534, -0.30151135,     //
                                           2.1105793, -0.30151135}},
    }));

}  // namespace
}  // namespace per_image_standardization
}  // namespace ml_adj
