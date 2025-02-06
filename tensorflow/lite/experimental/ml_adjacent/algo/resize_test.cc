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
#include "tensorflow/lite/experimental/ml_adjacent/algo/resize.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

using ::ml_adj::algo::Algo;
using ::ml_adj::data::OwningVectorRef;

namespace ml_adj {
namespace resize {
namespace {

struct ResizeTestParams {
  const std::vector<dim_t> img_dims;
  const std::vector<float> img_data;
  const std::vector<dim_t> size;
  const std::vector<float> expected_data;
  const std::vector<dim_t> expected_shape;
};

class ResizeTest : public ::testing::TestWithParam<ResizeTestParams> {};

TEST_P(ResizeTest, FloatPixelType) {
  constexpr float kAbsError = 0.01f;
  const ResizeTestParams& params = GetParam();

  // Image input.
  OwningVectorRef img(etype_t::f32);
  img.Resize(dims_t(params.img_dims));
  ASSERT_EQ(img.Bytes(), params.img_data.size() * sizeof(float));
  std::memcpy(img.Data(), params.img_data.data(), img.Bytes());

  // New size input.
  OwningVectorRef size(etype_t::i32);
  size.Resize({2});
  ASSERT_EQ(size.Bytes(), params.size.size() * sizeof(int));
  std::memcpy(size.Data(), params.size.data(), size.Bytes());

  // Empty output image.
  OwningVectorRef output(etype_t::f32);

  // Run image resize custom call.
  const Algo* resize = Impl_Resize();
  resize->process({&img, &size}, {&output});

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
    ResizeTests, ResizeTest,
    testing::ValuesIn({
        // 4-D Tensor of shape [batch, height, width, channels] is used below.

        // Check 2x2 image resizing to 4x4 size.
        ResizeTestParams{/*img_dims=*/{1, 2, 2, 1},
                         /*img_data=*/
                         {1, 2,  //
                          3, 4},
                         /*new_shape=*/{4, 4},
                         /*expected_data=*/
                         {1, 1.5, 2, 2,  //
                          2, 2.5, 3, 3,  //
                          3, 3.5, 4, 4,  //
                          3, 3.5, 4, 4},
                         /*expected_shape=*/{1, 4, 4, 1}},

        // Check vertical resizing of 2-element vector.
        ResizeTestParams{/*img_dims=*/{1, 2, 1, 1},
                         /*img_data=*/{3, 9},
                         /*size=*/{3, 1},
                         /*expected_data=*/{3, 7, 9},
                         /*expected_shape=*/{1, 3, 1, 1}},

        // Check horizontal resizing of 2-element vector.
        ResizeTestParams{/*img_dims=*/{1, 1, 2, 1},
                         /*img_data=*/{3, 6},
                         /*size=*/{1, 3},
                         /*expected_data=*/{3, 5, 6},
                         /*expected_shape=*/{1, 1, 3, 1}},

        // Check 2x2 image resizing.
        ResizeTestParams{/*img_dims=*/{1, 2, 2, 1},
                         /*img_data=*/
                         {3, 6,  //
                          9, 12},
                         /*size=*/{3, 3},
                         /*expected_data=*/
                         {3, 5, 6,   //
                          7, 9, 10,  //
                          9, 11, 12},
                         /*expected_shape=*/{1, 3, 3, 1}},

        // Check 2x2 image resizing (2 images in the batch).
        ResizeTestParams{/*img_dims=*/{2, 2, 2, 1},
                         /*img_data=*/
                         {3, 6,   //
                          9, 12,  //
                                  //
                          4, 10,  //
                          10, 16},
                         /*size=*/{3, 3},
                         /*expected_data=*/
                         {3, 5, 6,    //
                          7, 9, 10,   //
                          9, 11, 12,  //
                                      //
                          4, 8, 10,   //
                          8, 12, 14,  //
                          10, 14, 16},
                         /*expected_shape=*/{2, 3, 3, 1}},

        // Check 2x2 with 2 channels image resizing.
        ResizeTestParams{
            /*img_dims=*/{1, 2, 2, 2},
            /*img_data=*/{3, 4, 6, 10, 9, 10, 12, 16},
            /*size=*/{3, 3},
            /*expected_data=*/
            {3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16},
            /*expected_shape=*/{1, 3, 3, 2}},

        // Check horizontal resizing of 2-element vector with large values.
        ResizeTestParams{/*img_dims=*/{1, 1, 2, 1},
                         /*img_data=*/{32765, 32767},
                         /*size=*/{1, 3},
                         /*expected_data=*/{32765, 32766.33f, 32767},
                         /*expected_shape=*/{1, 1, 3, 1}},
    }));

}  // namespace
}  // namespace resize
}  // namespace ml_adj
