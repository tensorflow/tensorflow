/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/graph_transformations/lstm_utils.h"

#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
std::vector<testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5) {
  std::vector<testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}
}  // namespace

class CopyArrayDataTest : public ::testing::Test {
 public:
  CopyArrayDataTest() {}

  void PrepareBuffers(Model* model, std::initializer_list<float> src_data,
                      int src_dim_1, int src_dim_2,
                      std::initializer_list<float> dst_data, int dst_dim_1,
                      int dst_dim_2) {
    std::string src_array = "src_array";
    src_buffer_ = CreateFloatArrayBuffer(
        model, &src_array,
        src_dim_2 == 1 ? Shape({src_dim_1}) : Shape({src_dim_1, src_dim_2}));
    PopulateBuffer(src_buffer_, src_data);
    std::string dst_array = "dst_array";
    dst_buffer_ = CreateFloatArrayBuffer(
        model, &dst_array,
        dst_dim_2 == 1 ? Shape({dst_dim_1}) : Shape({dst_dim_1, dst_dim_2}));
    PopulateBuffer(dst_buffer_, dst_data);
  }

  Buffer<ArrayDataType::kFloat>* GetSrcBuffer() { return src_buffer_; }
  Buffer<ArrayDataType::kFloat>* GetDstBuffer() { return dst_buffer_; }

  void PopulateBuffer(Buffer<ArrayDataType::kFloat>* buffer,
                      const std::vector<float>& init_data) {
    for (int i = 0; i < init_data.size(); i++) {
      buffer->data[i] = init_data[i];
    }
  }
  void UpdateBuffer(Buffer<ArrayDataType::kFloat>* buffer,
                    std::initializer_list<float> data) {
    buffer->data.resize(data.size());
    PopulateBuffer(buffer, data);
  }

 private:
  Buffer<ArrayDataType::kFloat>* src_buffer_;
  Buffer<ArrayDataType::kFloat>* dst_buffer_;
};

// Copy from 1 big 2D array to 8 smaller ones.
TEST_F(CopyArrayDataTest, CopyFromBigArrayToSmallerArrayes2D) {
  // Init src_buffer, dst_buffer.
  Model model;
  std::initializer_list<float> large_tf_weight_data = {
      -0.320407, -0.108683, 0.406358,  -0.410811, -0.285786, -0.15769,
      -0.194201, 0.170866,  0.084135,  0.201878,  0.21519,   -0.284458,
      0.495906,  -0.073818, 0.045578,  0.149816,  -0.447073, -0.453578,
      0.116766,  0.21808,   0.047326,  -0.001985, 0.402193,  0.315517,
      0.38258,   0.43599,   0.11986,   0.465195,  0.33548,   -0.118789,
      -0.414159, 0.049269,  0.156108,  0.093459,  -0.129103, -0.086274,
      0.186188,  -0.324923, 0.4117,    -0.344439, 0.240465,  -0.343331,
      -0.463082, -0.231706, -0.487465, -0.186592, -0.020756, -0.239007,
      0.364817,  0.459106,  -0.171447, -0.006542, 0.204032,  -0.375317,
      -0.041911, 0.051664,  0.320483,  0.155899,  0.156555,  -0.249823,
      -0.353107, 0.031563,  -0.340771, -0.052532, 0.134631,  -0.257957,
      -0.50141,  0.486939,  -0.43853,  0.268426,  -0.08754,  -0.109447,
      -0.502462, -0.028055, -0.121838, -0.046016, 0.105309,  -0.070774,
      0.495683,  -0.475088, 0.048654,  -0.38582,  0.411018,  -0.315606,
      0.349628,  0.21698,   0.258989,  -0.097902, 0.331218,  0.034602,
      0.418069,  -0.089025, -0.417513, 0.07609,   0.393821,  0.404733,
      -0.055418, -0.43903,  -0.447049, 0.013125,  0.278503,  0.459869,
      0.143755,  -0.177335, -0.162247, -0.432371, 0.153714,  -0.047403,
      -0.446775, -0.418363, 0.019743,  0.042025};
  std::initializer_list<float> tflite_lstm_input_weight = {0, 0, 0, 0, 0, 0,
                                                           0, 0, 0, 0, 0, 0};
  PrepareBuffers(&model, large_tf_weight_data, /*src_dim_1=*/16,
                 /*src_dim_2=*/7, tflite_lstm_input_weight,
                 /*dst_dim_1=*/4, /*dst_dim_2=*/3);

  // Copy src starts at (0,0), size (4,3).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/3,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);
  std::vector<float> expected = {-0.320407, -0.108683, 0.406358, 0.170866,
                                 0.084135,  0.201878,  0.045578, 0.149816,
                                 -0.447073, -0.001985, 0.402193, 0.315517};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy src starts at (4,0), size (4,3).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/4,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/3,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);
  expected = {0.33548,   -0.118789, -0.414159, -0.086274, 0.186188,  -0.324923,
              -0.463082, -0.231706, -0.487465, 0.459106,  -0.171447, -0.006542};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy src starts at (8,0), size (4,3).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/8,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/3,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);
  expected = {0.320483, 0.155899,  0.156555,  -0.052532, 0.134631, -0.257957,
              -0.08754, -0.109447, -0.502462, -0.070774, 0.495683, -0.475088};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy src starts at (12,0), size (4,3).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/12,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/3,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);
  expected = {0.349628,  0.21698,  0.258989, -0.089025, -0.417513, 0.07609,
              -0.447049, 0.013125, 0.278503, -0.432371, 0.153714,  -0.047403};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // New dst_buffer with size 16.
  std::initializer_list<float> tflite_lstm_recurrent_weight = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  PrepareBuffers(&model, large_tf_weight_data, /*src_dim_1=*/16,
                 /*src_dim_2=*/7, tflite_lstm_recurrent_weight,
                 /*dst_dim_1=*/4, /*dst_dim_2=*/4);

  // Copy src starts at (0,3), size (4,4).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/0,
                /*src_start_idx2=*/3, GetDstBuffer(), /*dst_stride=*/4,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);
  expected = {-0.410811, -0.285786, -0.15769,  -0.194201, 0.21519, -0.284458,
              0.495906,  -0.073818, -0.453578, 0.116766,  0.21808, 0.047326,
              0.38258,   0.43599,   0.11986,   0.465195};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy src starts at (4,3), size (4,4).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/4,
                /*src_start_idx2=*/3, GetDstBuffer(), /*dst_stride=*/4,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);
  expected = {0.049269, 0.156108,  0.093459,  -0.129103, 0.4117,    -0.344439,
              0.240465, -0.343331, -0.186592, -0.020756, -0.239007, 0.364817,
              0.204032, -0.375317, -0.041911, 0.051664};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy src starts at (8,3), size (4,4).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/8,
                /*src_start_idx2=*/3, GetDstBuffer(), /*dst_stride=*/4,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);
  expected = {-0.249823, -0.353107, 0.031563,  -0.340771, -0.50141,  0.486939,
              -0.43853,  0.268426,  -0.028055, -0.121838, -0.046016, 0.105309,
              0.048654,  -0.38582,  0.411018,  -0.315606};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy src starts at (12,3), size (4,4).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/7, /*src_start_idx1=*/12,
                /*src_start_idx2=*/3, GetDstBuffer(), /*dst_stride=*/4,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);
  expected = {-0.097902, 0.331218,  0.034602, 0.418069, 0.393821,  0.404733,
              -0.055418, -0.43903,  0.459869, 0.143755, -0.177335, -0.162247,
              -0.446775, -0.418363, 0.019743, 0.042025};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));
}

// Copy from 1 big 1D array to 4 small ones.
TEST_F(CopyArrayDataTest, CopyFromBigArrayToSmallerArrayes1D) {
  // Init src_buffer, dst_buffer.
  Model model;
  std::initializer_list<float> large_tf_bias_data = {
      0.980304, 0.419808, 0.080278, 0.728548, 0.581674, 0.672433,
      0.434190, 0.844357, 0.229587, 0.785629, 0.022065, 0.753082,
      0.422080, 0.539481, 0.878386, 0.168965};
  std::initializer_list<float> tflite_lstm_i_bias = {0, 0, 0, 0};
  PrepareBuffers(&model, large_tf_bias_data, /*src_dim_1=*/16,
                 /*src_dim_2=*/1, tflite_lstm_i_bias,
                 /*dst_dim_1=*/4, /*dst_dim_2=*/1);

  // Copy starts at (0,), size (4,).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);
  std::vector<float> expected = {0.980304, 0.419808, 0.080278, 0.728548};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy starts at (4,), size (4,).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/4,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);
  expected = {0.581674, 0.672433, 0.434190, 0.844357};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy starts at (8,), size (4,).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/8,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);
  expected = {0.229587, 0.785629, 0.022065, 0.753082};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));

  // Copy starts at (12,), size (4,).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/12,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);
  expected = {0.422080, 0.539481, 0.878386, 0.168965};
  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));
}

// Copy from 8 small 2D arrayes to 1 big one.
TEST_F(CopyArrayDataTest, CopyFromSmallArrayesToBigArray2D) {
  // Init src_buffer, dst_buffer.
  Model model;
  std::initializer_list<float> large_tf_weights_data = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Copy dst starts (0, 0), size (4, 3).
  std::initializer_list<float> tflite_lstm_i2i_weight = {
      -0.320407, -0.108683, 0.406358,  0.170866,  0.084135, 0.201878,
      0.045578,  0.149816,  -0.447073, -0.001985, 0.402193, 0.315517};
  PrepareBuffers(&model, tflite_lstm_i2i_weight, /*src_dim_1=*/4,
                 /*src_dim_2=*/3, large_tf_weights_data,
                 /*dst_dim_1=*/16, /*dst_dim_2=*/7);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/3, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);

  // Copy dst starts (4, 0), size (4, 3).
  std::initializer_list<float> tflite_lstm_i2c_weight = {
      0.33548,   -0.118789, -0.414159, -0.086274, 0.186188,  -0.324923,
      -0.463082, -0.231706, -0.487465, 0.459106,  -0.171447, -0.006542};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_i2c_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/3, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/4, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);

  // Copy dst starts (8, 0), size (4, 3).
  std::initializer_list<float> tflite_lstm_i2f_weight = {
      0.320483, 0.155899,  0.156555,  -0.052532, 0.134631, -0.257957,
      -0.08754, -0.109447, -0.502462, -0.070774, 0.495683, -0.475088};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_i2f_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/3, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/8, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);

  // Copy dst starts (12, 0), size (4, 3).
  std::initializer_list<float> tflite_lstm_i2o_weight = {
      0.349628,  0.21698,  0.258989, -0.089025, -0.417513, 0.07609,
      -0.447049, 0.013125, 0.278503, -0.432371, 0.153714,  -0.047403};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_i2o_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/3, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/12, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/3);

  // Copy dst starts (0, 3), size (4, 4).
  std::initializer_list<float> tflite_lstm_i2r_weight = {
      -0.410811, -0.285786, -0.15769,  -0.194201, 0.21519, -0.284458,
      0.495906,  -0.073818, -0.453578, 0.116766,  0.21808, 0.047326,
      0.38258,   0.43599,   0.11986,   0.465195};
  UpdateBuffer(GetSrcBuffer(), tflite_lstm_i2r_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/4, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/3,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);

  // Copy dst starts (4, 3), size (4, 4).
  std::initializer_list<float> tflite_lstm_c2r_weight = {
      0.049269, 0.156108,  0.093459,  -0.129103, 0.4117,    -0.344439,
      0.240465, -0.343331, -0.186592, -0.020756, -0.239007, 0.364817,
      0.204032, -0.375317, -0.041911, 0.051664};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_c2r_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/4, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/4, /*dst_start_idx2=*/3,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);

  // Copy dst starts (8, 3), size (4, 4).
  std::initializer_list<float> tflite_lstm_f2r_weight = {
      -0.249823, -0.353107, 0.031563,  -0.340771, -0.50141,  0.486939,
      -0.43853,  0.268426,  -0.028055, -0.121838, -0.046016, 0.105309,
      0.048654,  -0.38582,  0.411018,  -0.315606};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_f2r_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/4, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/8, /*dst_start_idx2=*/3,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);

  // Copy dst starts (12, 3), size (4, 4).
  std::initializer_list<float> tflite_lstm_o2r_weight = {
      -0.097902, 0.331218,  0.034602, 0.418069, 0.393821,  0.404733,
      -0.055418, -0.43903,  0.459869, 0.143755, -0.177335, -0.162247,
      -0.446775, -0.418363, 0.019743, 0.042025};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_o2r_weight);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/4, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/7,
                /*dst_start_idx1=*/12, /*dst_start_idx2=*/3,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/4);

  std::vector<float> expected = {
      -0.320407, -0.108683, 0.406358,  -0.410811, -0.285786, -0.15769,
      -0.194201, 0.170866,  0.084135,  0.201878,  0.21519,   -0.284458,
      0.495906,  -0.073818, 0.045578,  0.149816,  -0.447073, -0.453578,
      0.116766,  0.21808,   0.047326,  -0.001985, 0.402193,  0.315517,
      0.38258,   0.43599,   0.11986,   0.465195,  0.33548,   -0.118789,
      -0.414159, 0.049269,  0.156108,  0.093459,  -0.129103, -0.086274,
      0.186188,  -0.324923, 0.4117,    -0.344439, 0.240465,  -0.343331,
      -0.463082, -0.231706, -0.487465, -0.186592, -0.020756, -0.239007,
      0.364817,  0.459106,  -0.171447, -0.006542, 0.204032,  -0.375317,
      -0.041911, 0.051664,  0.320483,  0.155899,  0.156555,  -0.249823,
      -0.353107, 0.031563,  -0.340771, -0.052532, 0.134631,  -0.257957,
      -0.50141,  0.486939,  -0.43853,  0.268426,  -0.08754,  -0.109447,
      -0.502462, -0.028055, -0.121838, -0.046016, 0.105309,  -0.070774,
      0.495683,  -0.475088, 0.048654,  -0.38582,  0.411018,  -0.315606,
      0.349628,  0.21698,   0.258989,  -0.097902, 0.331218,  0.034602,
      0.418069,  -0.089025, -0.417513, 0.07609,   0.393821,  0.404733,
      -0.055418, -0.43903,  -0.447049, 0.013125,  0.278503,  0.459869,
      0.143755,  -0.177335, -0.162247, -0.432371, 0.153714,  -0.047403,
      -0.446775, -0.418363, 0.019743,  0.042025};

  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));
}

// Copy from 4 small 1D arrayes to 1 big one.
TEST_F(CopyArrayDataTest, CopyFromSmallArrayesToBigArray1D) {
  // Init src_buffer, dst_buffer.
  Model model;
  std::initializer_list<float> large_tf_bias_data = {0, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0};

  std::initializer_list<float> tflite_lstm_i_bias = {0.980304, 0.419808,
                                                     0.080278, 0.728548};

  PrepareBuffers(&model, tflite_lstm_i_bias, /*src_dim_1=*/4,
                 /*src_dim_2=*/1, large_tf_bias_data,
                 /*dst_dim_1=*/16, /*dst_dim_2=*/1);

  // Copy starts at (0,), size (4,).
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/0, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);

  // Copy starts at (4,), size (4,).
  std::initializer_list<float> tflite_lstm_cell_bias = {0.581674, 0.672433,
                                                        0.434190, 0.844357};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_cell_bias);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/4, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);

  // Copy starts at (8,0), size (4,).
  std::initializer_list<float> tflite_lstm_forget_bias = {0.229587, 0.785629,
                                                          0.022065, 0.753082};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_forget_bias);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/8, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);

  // Copy starts at (12,), size (4,).
  std::initializer_list<float> tflite_lstm_output_bias = {0.422080, 0.539481,
                                                          0.878386, 0.168965};
  PopulateBuffer(GetSrcBuffer(), tflite_lstm_output_bias);
  CopyArrayData(*(GetSrcBuffer()),
                /*src_stride=*/1, /*src_start_idx1=*/0,
                /*src_start_idx2=*/0, GetDstBuffer(), /*dst_stride=*/1,
                /*dst_start_idx1=*/12, /*dst_start_idx2=*/0,
                /*dim1_copy_size=*/4, /*dim2_copy_size=*/1);

  std::vector<float> expected = {0.980304, 0.419808, 0.080278, 0.728548,
                                 0.581674, 0.672433, 0.434190, 0.844357,
                                 0.229587, 0.785629, 0.022065, 0.753082,
                                 0.422080, 0.539481, 0.878386, 0.168965};

  EXPECT_THAT(GetDstBuffer()->data, ElementsAreArray(ArrayFloatNear(expected)));
}

}  // namespace toco
