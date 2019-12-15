/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/lstm_eval.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

// Validate result.
template <typename T>
bool ArrayEq(const T* result, const T* expected_result, int size) {
  for (int i = 0; i < size; ++i) {
    if (result[i] != expected_result[i]) {
      return false;
    }
  }
  return true;
}

// The class that holds input parameters for quantized lstm.
class QuantizedLstmParam {
 public:
  // Getter methods.
  TfLiteTensor* GetInput() {
    PackWeightToTensor(&input_tensor_, input_, input_size_);
    input_tensor_.data.int8 = input_.data();
    return &input_tensor_;
  }
  TfLiteTensor* Geti2i() {
    PackWeightToTensor(&i2i_tensor_, i2i_, i2i_size_);
    i2i_tensor_.data.int8 = i2i_.data();
    return &i2i_tensor_;
  }
  TfLiteTensor* Geti2f() {
    PackWeightToTensor(&i2f_tensor_, i2f_, i2f_size_);
    i2f_tensor_.data.int8 = i2f_.data();
    return &i2f_tensor_;
  }
  TfLiteTensor* Geti2c() {
    PackWeightToTensor(&i2c_tensor_, i2c_, i2c_size_);
    i2c_tensor_.data.int8 = i2c_.data();
    return &i2c_tensor_;
  }
  TfLiteTensor* Geti2o() {
    PackWeightToTensor(&i2o_tensor_, i2o_, i2o_size_);
    i2o_tensor_.data.int8 = i2o_.data();
    return &i2o_tensor_;
  }
  TfLiteTensor* Getr2i() {
    PackWeightToTensor(&r2i_tensor_, r2i_, r2i_size_);
    r2i_tensor_.data.int8 = r2i_.data();
    return &r2i_tensor_;
  }
  TfLiteTensor* Getr2f() {
    PackWeightToTensor(&r2f_tensor_, r2f_, r2f_size_);
    r2f_tensor_.data.int8 = r2f_.data();
    return &r2f_tensor_;
  }
  TfLiteTensor* Getr2c() {
    PackWeightToTensor(&r2c_tensor_, r2c_, r2c_size_);
    r2c_tensor_.data.int8 = r2c_.data();
    return &r2c_tensor_;
  }
  TfLiteTensor* Getr2o() {
    PackWeightToTensor(&r2o_tensor_, r2o_, r2o_size_);
    r2o_tensor_.data.int8 = r2o_.data();
    return &r2o_tensor_;
  }
  TfLiteTensor* GetProjection() {
    PackWeightToTensor(&projection_tensor_, projection_, projection_size_);
    projection_tensor_.data.int8 = projection_.data();
    return &projection_tensor_;
  }
  TfLiteTensor* GetInputLayerNorm() {
    PackWeightToTensor(&layer_norm_input_tensor_, layer_norm_input_,
                       layer_norm_input_size_);
    layer_norm_input_tensor_.data.i16 = layer_norm_input_.data();
    return &layer_norm_input_tensor_;
  }
  TfLiteTensor* GetForgetLayerNorm() {
    PackWeightToTensor(&layer_norm_forget_tensor_, layer_norm_forget_,
                       layer_norm_forget_size_);
    layer_norm_forget_tensor_.data.i16 = layer_norm_forget_.data();
    return &layer_norm_forget_tensor_;
  }
  TfLiteTensor* GetCellLayerNorm() {
    PackWeightToTensor(&layer_norm_cell_tensor_, layer_norm_cell_,
                       layer_norm_cell_size_);
    layer_norm_cell_tensor_.data.i16 = layer_norm_cell_.data();
    return &layer_norm_cell_tensor_;
  }
  TfLiteTensor* GetOutputLayerNorm() {
    PackWeightToTensor(&layer_norm_output_tensor_, layer_norm_output_,
                       layer_norm_output_size_);
    layer_norm_output_tensor_.data.i16 = layer_norm_output_.data();
    return &layer_norm_output_tensor_;
  }
  TfLiteTensor* GetInputBias() {
    PackWeightToTensor(&input_bias_tensor_, input_bias_, input_bias_size_);
    input_bias_tensor_.data.i32 = input_bias_.data();
    return &input_bias_tensor_;
  }
  TfLiteTensor* GetForgetBias() {
    PackWeightToTensor(&forget_bias_tensor_, forget_bias_, forget_bias_size_);
    forget_bias_tensor_.data.i32 = forget_bias_.data();
    return &forget_bias_tensor_;
  }
  TfLiteTensor* GetCellBias() {
    PackWeightToTensor(&cell_bias_tensor_, cell_bias_, cell_bias_size_);
    cell_bias_tensor_.data.i32 = cell_bias_.data();
    return &cell_bias_tensor_;
  }
  TfLiteTensor* GetOutputBias() {
    PackWeightToTensor(&output_bias_tensor_, output_bias_, output_bias_size_);
    output_bias_tensor_.data.i32 = output_bias_.data();
    return &output_bias_tensor_;
  }
  TfLiteTensor* GetProjectionBias() {
    PackWeightToTensor(&projection_bias_tensor_, projection_bias_,
                       projection_bias_size_);
    projection_bias_tensor_.data.i32 = projection_bias_.data();
    return &projection_bias_tensor_;
  }

  // Set up quantization parameters.
  ops::builtin::lstm_eval::QuantizedLstmParameter* GetQuantParam() {
    quant_lstm_parm_.effective_input_to_input_scale_a = 1808677632;
    quant_lstm_parm_.effective_input_to_input_scale_b = -1;
    quant_lstm_parm_.effective_recurrent_to_input_scale_a = 1078887680;
    quant_lstm_parm_.effective_recurrent_to_input_scale_b = -1;
    quant_lstm_parm_.effective_cell_to_input_scale_a = 1073741824;
    quant_lstm_parm_.effective_cell_to_input_scale_b = 1;
    quant_lstm_parm_.effective_input_to_forget_scale_a = 1845996800;
    quant_lstm_parm_.effective_input_to_forget_scale_b = -3;
    quant_lstm_parm_.effective_recurrent_to_forget_scale_a = 1477412736;
    quant_lstm_parm_.effective_recurrent_to_forget_scale_b = -2;
    quant_lstm_parm_.effective_cell_to_forget_scale_a = 1073741824;
    quant_lstm_parm_.effective_cell_to_forget_scale_b = 1;
    quant_lstm_parm_.effective_input_to_cell_scale_a = 1648385408;
    quant_lstm_parm_.effective_input_to_cell_scale_b = -2;
    quant_lstm_parm_.effective_recurrent_to_cell_scale_a = 1185544192,
    quant_lstm_parm_.effective_recurrent_to_cell_scale_b = -1;
    quant_lstm_parm_.effective_input_to_output_scale_a = 1328153600;
    quant_lstm_parm_.effective_input_to_output_scale_b = -1;
    quant_lstm_parm_.effective_recurrent_to_output_scale_a = 1479582592;
    quant_lstm_parm_.effective_recurrent_to_output_scale_b = -1;
    quant_lstm_parm_.effective_cell_to_output_scale_a = 1073741824,
    quant_lstm_parm_.effective_cell_to_output_scale_b = 1;
    quant_lstm_parm_.effective_proj_scale_a = 1105682560;
    quant_lstm_parm_.effective_proj_scale_b = -8;
    quant_lstm_parm_.effective_hidden_scale_a = 0;
    quant_lstm_parm_.effective_hidden_scale_b = 0;
    quant_lstm_parm_.layer_norm_input_scale_a = 2011617664;
    quant_lstm_parm_.layer_norm_input_scale_b = -11;
    quant_lstm_parm_.layer_norm_forget_scale_a = 1968024960;
    quant_lstm_parm_.layer_norm_forget_scale_b = -13;
    quant_lstm_parm_.layer_norm_cell_scale_a = 1097334528,
    quant_lstm_parm_.layer_norm_cell_scale_b = -12;
    quant_lstm_parm_.layer_norm_output_scale_a = 1837163008;
    quant_lstm_parm_.layer_norm_output_scale_b = -12;
    quant_lstm_parm_.quantized_cell_clip = 20480;
    quant_lstm_parm_.quantized_proj_clip = 0;
    quant_lstm_parm_.cell_scale = -11;
    quant_lstm_parm_.inv_large_value[0] = 1;
    quant_lstm_parm_.inv_large_value[1] = 2;
    quant_lstm_parm_.inv_large_value[2] = 2;
    quant_lstm_parm_.inv_large_value[3] = 1;
    quant_lstm_parm_.hidden_zp = 0;
    quant_lstm_parm_.input_to_forget_effective_bias.reset(new int32_t[n_cell_]);
    quant_lstm_parm_.recurrent_to_forget_effective_bias.reset(
        new int32_t[n_cell_]);
    quant_lstm_parm_.input_to_cell_effective_bias.reset(new int32_t[n_cell_]);
    quant_lstm_parm_.recurrent_to_cell_effective_bias.reset(
        new int32_t[n_cell_]);
    quant_lstm_parm_.input_to_output_effective_bias.reset(new int32_t[n_cell_]);
    quant_lstm_parm_.recurrent_to_output_effective_bias.reset(
        new int32_t[n_cell_]);
    quant_lstm_parm_.input_to_input_effective_bias.reset(new int32_t[n_cell_]);
    quant_lstm_parm_.recurrent_to_input_effective_bias.reset(
        new int32_t[n_cell_]);
    quant_lstm_parm_.projection_effective_bias.reset(new int32_t[n_output_]);
    std::fill_n(quant_lstm_parm_.input_to_forget_effective_bias.get(), n_cell_,
                152);
    std::fill_n(quant_lstm_parm_.recurrent_to_forget_effective_bias.get(),
                n_cell_, 315);
    std::fill_n(quant_lstm_parm_.input_to_cell_effective_bias.get(), n_cell_,
                165);
    std::fill_n(quant_lstm_parm_.recurrent_to_cell_effective_bias.get(),
                n_cell_, 1165);
    std::fill_n(quant_lstm_parm_.input_to_output_effective_bias.get(), n_cell_,
                159);
    std::fill_n(quant_lstm_parm_.recurrent_to_output_effective_bias.get(),
                n_cell_, 915);
    std::fill_n(quant_lstm_parm_.input_to_input_effective_bias.get(), n_cell_,
                -15);
    std::fill_n(quant_lstm_parm_.recurrent_to_input_effective_bias.get(),
                n_cell_, 315);
    std::fill_n(quant_lstm_parm_.projection_effective_bias.get(), n_output_,
                115);
    return &quant_lstm_parm_;
  }

  // Create scratch buffers.
  TfLiteTensor* GetScratch0() {
    PackWeightToTensor(&scratch0_tensor_, scratch0_, scratch0_size_);
    scratch0_tensor_.data.i16 = scratch0_.data();
    return &scratch0_tensor_;
  }
  TfLiteTensor* GetScratch1() {
    PackWeightToTensor(&scratch1_tensor_, scratch1_, scratch1_size_);
    scratch1_tensor_.data.i16 = scratch1_.data();
    return &scratch1_tensor_;
  }
  TfLiteTensor* GetScratch2() {
    PackWeightToTensor(&scratch2_tensor_, scratch2_, scratch2_size_);
    scratch2_tensor_.data.i16 = scratch2_.data();
    return &scratch2_tensor_;
  }
  TfLiteTensor* GetScratch3() {
    PackWeightToTensor(&scratch3_tensor_, scratch3_, scratch3_size_);
    scratch3_tensor_.data.i16 = scratch3_.data();
    return &scratch3_tensor_;
  }
  TfLiteTensor* GetScratch4() {
    PackWeightToTensor(&scratch4_tensor_, scratch4_, scratch4_size_);
    scratch4_tensor_.data.int8 = scratch4_.data();
    return &scratch4_tensor_;
  }
  TfLiteTensor* GetScratch5() {
    PackWeightToTensor(&scratch5_tensor_, scratch5_, scratch5_size_);
    scratch5_tensor_.data.i32 = scratch5_.data();
    return &scratch5_tensor_;
  }
  TfLiteTensor* GetActivation() {
    PackWeightToTensor(&activation_tensor_, activation_, activation_size_);
    activation_tensor_.data.int8 = activation_.data();
    activation_tensor_.params.zero_point = 50;
    return &activation_tensor_;
  }
  TfLiteTensor* GetOutput() {
    PackWeightToTensor(&output_tensor_, output_, output_size_);
    output_tensor_.data.int8 = output_.data();
    return &output_tensor_;
  }
  TfLiteTensor* GetCell() {
    PackWeightToTensor(&cell_tensor_, cell_, cell_size_);
    cell_tensor_.data.i16 = cell_.data();
    return &cell_tensor_;
  }

  ~QuantizedLstmParam() {
    TfLiteIntArrayFree(input_tensor_.dims);
    TfLiteIntArrayFree(i2i_tensor_.dims);
    TfLiteIntArrayFree(i2f_tensor_.dims);
    TfLiteIntArrayFree(i2c_tensor_.dims);
    TfLiteIntArrayFree(i2o_tensor_.dims);
    TfLiteIntArrayFree(r2i_tensor_.dims);
    TfLiteIntArrayFree(r2f_tensor_.dims);
    TfLiteIntArrayFree(r2c_tensor_.dims);
    TfLiteIntArrayFree(r2o_tensor_.dims);
    TfLiteIntArrayFree(projection_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_input_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_forget_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_cell_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_output_tensor_.dims);
    TfLiteIntArrayFree(input_bias_tensor_.dims);
    TfLiteIntArrayFree(forget_bias_tensor_.dims);
    TfLiteIntArrayFree(cell_bias_tensor_.dims);
    TfLiteIntArrayFree(output_bias_tensor_.dims);
    TfLiteIntArrayFree(projection_bias_tensor_.dims);
    TfLiteIntArrayFree(activation_tensor_.dims);
    TfLiteIntArrayFree(cell_tensor_.dims);
    TfLiteIntArrayFree(output_tensor_.dims);
    TfLiteIntArrayFree(scratch0_tensor_.dims);
    TfLiteIntArrayFree(scratch1_tensor_.dims);
    TfLiteIntArrayFree(scratch2_tensor_.dims);
    TfLiteIntArrayFree(scratch3_tensor_.dims);
    TfLiteIntArrayFree(scratch4_tensor_.dims);
    TfLiteIntArrayFree(scratch5_tensor_.dims);
  }

 private:
  template <typename T>
  void PackWeightToTensor(TfLiteTensor* tensor, std::vector<T>& data,
                          std::vector<int32_t> dims) {
    if (data.empty()) {
      int total = 1;
      for (int i = 0; i < dims.size(); ++i) {
        total *= dims[i];
      }
      for (int i = 0; i < total; ++i) {
        data.push_back(0);
      }
    }
    tensor->dims = TfLiteIntArrayCreate(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      tensor->dims->data[i] = dims[i];
    }
  }

  // Dimensions. Need proper size to trigger neon code.
  const int n_batch_ = 2;
  const int n_input_ = 18;
  const int n_cell_ = 10;
  const int n_output_ = 6;
  // input.
  std::vector<int8_t> input_ = {
      8, 2, 3,  4, 5, 6, 1, -2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,  //
      1, 2, -3, 4, 5, 6, 1, 2,  3, 4, 5, 6, 1, 2, 3, 4, 5, 6,  //
  };
  std::vector<int32_t> input_size_ = {n_batch_, n_input_};
  TfLiteTensor input_tensor_;

  // input_to_input_weights.
  std::vector<int8_t> i2i_ = {
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  0,   //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6, 1, 2, 3, -4, 5,  6,   //
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6, 1, 7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 8,  5,  -6,  //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6, 1, 2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  6, 1, 2, 3, 14, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
  };
  std::vector<int32_t> i2i_size_ = {n_cell_, n_input_};
  TfLiteTensor i2i_tensor_;

  // input_to_forget_weights.
  std::vector<int8_t> i2f_ = {
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  0,   //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1,  2, 3, -4, 5,  6,   //
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1,  7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  11, 2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  -6, 1,  2, 3, 14, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  13, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 8,  5,  -6,  //
  };
  std::vector<int32_t> i2f_size_ = {n_cell_, n_input_};
  TfLiteTensor i2f_tensor_;
  // input_to_cell_weights.
  std::vector<int8_t> i2c_ = {
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  0,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  1, 2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  16, 1, 2, 3, 14, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  7, 2, 3, 4,  5,  6,   //
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 8,  5,  -6,  //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1, 2, 3, -4, 5,  6,   //
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1, 7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
  };
  std::vector<int32_t> i2c_size_ = {n_cell_, n_input_};
  TfLiteTensor i2c_tensor_;

  // input_to_output_weights.
  std::vector<int8_t> i2o_ = {
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1,  7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  -1, 2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  1,  2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  6,  1,  2, 3, 14, 5,  6,   //
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  -6, 1,  2, 3, 4,  5,  6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  0,   //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1,  2, 3, -4, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  -1, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 8,  5,  -6,  //
  };
  std::vector<int32_t> i2o_size_ = {n_cell_, n_input_};
  TfLiteTensor i2o_tensor_;

  // recurrent_to_input_weights.
  std::vector<int8_t> r2i_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2i_size_ = {n_cell_, n_output_};
  TfLiteTensor r2i_tensor_;

  // recurrent_to_forget_weights.
  std::vector<int8_t> r2f_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2f_size_ = {n_cell_, n_output_};
  TfLiteTensor r2f_tensor_;

  // recurrent_to_cell_weights.
  std::vector<int8_t> r2c_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2c_size_ = {n_cell_, n_output_};
  TfLiteTensor r2c_tensor_;

  // recurrent_to_output_weights.
  std::vector<int8_t> r2o_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2o_size_ = {n_cell_, n_output_};
  TfLiteTensor r2o_tensor_;

  // input_layer_norm_coefficients.
  std::vector<int16_t> layer_norm_input_ = {8, 2, 3, 4, 5, 6, 1, 2, 3, 4};
  std::vector<int32_t> layer_norm_input_size_ = {n_cell_};
  TfLiteTensor layer_norm_input_tensor_;

  // forget_layer_norm_coefficient.
  std::vector<int16_t> layer_norm_forget_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6, 3,  //
  };
  std::vector<int32_t> layer_norm_forget_size_ = {n_cell_};
  TfLiteTensor layer_norm_forget_tensor_;

  // cell_layer_norm_coefficients.
  std::vector<int16_t> layer_norm_cell_ = {
      6, 4, 5, 6, 1, 2, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> layer_norm_cell_size_ = {n_cell_};
  TfLiteTensor layer_norm_cell_tensor_;

  // output_layer_norm_coefficients.
  std::vector<int16_t> layer_norm_output_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> layer_norm_output_size_ = {n_cell_};
  TfLiteTensor layer_norm_output_tensor_;

  // input_gate_bias.
  std::vector<int32_t> input_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> input_bias_size_ = {n_cell_};
  TfLiteTensor input_bias_tensor_;

  // forget_gate_bias.
  std::vector<int32_t> forget_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> forget_bias_size_ = {n_cell_};
  TfLiteTensor forget_bias_tensor_;

  // cell_bias.
  std::vector<int32_t> cell_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> cell_bias_size_ = {n_cell_};
  TfLiteTensor cell_bias_tensor_;

  // output_gate_bias.
  std::vector<int32_t> output_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> output_bias_size_ = {n_cell_};
  TfLiteTensor output_bias_tensor_;

  // projection_weights.
  std::vector<int8_t> projection_ = {
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
  };
  std::vector<int32_t> projection_size_ = {n_cell_, n_output_};
  TfLiteTensor projection_tensor_;

  // projection_bias.
  std::vector<int32_t> projection_bias_ = {
      16, 4, 5, 6, 1, 1  //
  };
  std::vector<int32_t> projection_bias_size_ = {n_output_};
  TfLiteTensor projection_bias_tensor_;

  // activation.
  std::vector<int8_t> activation_;
  std::vector<int32_t> activation_size_ = {n_batch_, n_output_};
  TfLiteTensor activation_tensor_;

  // cell.
  std::vector<int16_t> cell_ = {
      16, 4,  5, 6, 1, 1, 3, 4, -5, 6,  //
      1,  14, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> cell_size_ = {n_batch_, n_cell_};
  TfLiteTensor cell_tensor_;

  // output.
  std::vector<int8_t> output_ = {
      1, 1, 3, 4, -5, 6,  //
      1, 4, 3, 4, -5, 6,  //
  };
  std::vector<int32_t> output_size_ = {n_batch_, n_output_};
  TfLiteTensor output_tensor_;

  // quantized_lstm_param
  ops::builtin::lstm_eval::QuantizedLstmParameter quant_lstm_parm_;

  // 5 scratch buffers.
  std::vector<int16_t> scratch0_;
  std::vector<int32_t> scratch0_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch0_tensor_;
  std::vector<int16_t> scratch1_;
  std::vector<int32_t> scratch1_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch1_tensor_;
  std::vector<int16_t> scratch2_;
  std::vector<int32_t> scratch2_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch2_tensor_;
  std::vector<int16_t> scratch3_;
  std::vector<int32_t> scratch3_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch3_tensor_;
  std::vector<int8_t> scratch4_;
  std::vector<int32_t> scratch4_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch4_tensor_;
  std::vector<int32_t> scratch5_;
  std::vector<int32_t> scratch5_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch5_tensor_;
};

void TestOneFullyQuantizedLSTM() {
  CpuBackendContext context;
  QuantizedLstmParam one_parameter;
  auto activation = one_parameter.GetActivation();
  auto output = one_parameter.GetOutput();
  auto cell = one_parameter.GetCell();
  auto param = one_parameter.GetQuantParam();
  ops::builtin::lstm_eval::EvalQuantized(
      one_parameter.GetInput(), one_parameter.Geti2i(), one_parameter.Geti2f(),
      one_parameter.Geti2c(), one_parameter.Geti2o(), one_parameter.Getr2i(),
      one_parameter.Getr2f(), one_parameter.Getr2c(), one_parameter.Getr2o(),
      nullptr, nullptr, nullptr, one_parameter.GetInputLayerNorm(),
      one_parameter.GetForgetLayerNorm(), one_parameter.GetCellLayerNorm(),
      one_parameter.GetOutputLayerNorm(), one_parameter.GetInputBias(),
      one_parameter.GetForgetBias(), one_parameter.GetCellBias(),
      one_parameter.GetOutputBias(), one_parameter.GetProjection(),
      one_parameter.GetProjectionBias(), nullptr, param, activation, cell,
      output, one_parameter.GetScratch0(), one_parameter.GetScratch1(),
      one_parameter.GetScratch2(), one_parameter.GetScratch3(),
      one_parameter.GetScratch4(), one_parameter.GetScratch5(), &context);

  // Verify results.
  const std::vector<int16_t> expected_cell = {
      7, 1, 3, 2, 0, 1, 0, 2, -2, 4, 1, 6, 4, 3, 0, 1, 0, 2, -2, 4,
  };
  const std::vector<int8_t> expected_activation = {
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
  };
  EXPECT_TRUE(ArrayEq(cell->data.i16, expected_cell.data(), 20));
  EXPECT_TRUE(ArrayEq(activation->data.int8, expected_activation.data(), 12));
  EXPECT_TRUE(ArrayEq(output->data.int8, expected_activation.data(), 12));
}

TEST(TestOneFullyQuantizedLSTM, TestOneFullyQuantizedLSTM) {
  TestOneFullyQuantizedLSTM();
}
}  // namespace
}  // namespace tflite
