/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Simple test data for 2x2x10 input 2x3x10 weights.
const int simple_input_size = 20;
const int simple_input_dims[] = {2, 2, 10};
const float simple_input_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
    1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
};
const int simple_weights_size = 30;
const int simple_weights_dims[] = {2, 3, 10};
const float simple_weights_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
};
const int simple_bias_dims[] = {1, 3};
const float simple_bias_data[] = {1, 2, 3};
const float simple_golden[] = {
    24, 25, 26, 58, 59, 60,
};
const int simple_output_size = 6;
const int simple_output_dims[] = {2, 2, 3};

// Test data for 2x2x10 input 2x3x10 weights with negative outputs to test relu.
const int relu_input_size = 20;
const int relu_input_dims[] = {2, 2, 10};
const float relu_input_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
    1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
};
const int relu_weights_size = 30;
const int relu_weights_dims[] = {2, 3, 10};
const float relu_weights_data[] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 0
    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,  // u = 1
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 2
};
const int relu_bias_dims[] = {1, 3};
const float relu_bias_data[] = {1, -2, 3};
const float relu_golden[] = {
    24, 0, 26, 58, 0, 60,
};
const int relu_output_size = 6;
const int relu_output_dims[] = {2, 2, 3};

// Input and filter similar to real model. Input shape is 1x64 and output is
// 1x16.
const int representative_64x16_input_size = 64;
const int representative_64x16_input_dims[] = {2, 1, 64};
const float representative_64x16_input_data[] = {
    0.0000, 0.1543, 0.0000, 0.0000, 1.8520, 0.0000, 4.7844, 1.1832,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.5948, 0.0000,
    1.5948, 1.9549, 0.0000, 1.2347, 0.0000, 1.5948, 1.5948, 0.5145,
    0.0000, 0.0000, 0.0000, 0.0000, 2.6237, 0.0000, 0.0000, 0.0000,
    1.3890, 5.3503, 2.3665, 2.9838, 0.0000, 1.2861, 0.0000, 3.0867,
    0.9775, 0.0000, 5.9676, 0.0000, 0.0000, 1.4405, 0.5145, 2.5723,
    3.1896, 4.4757, 0.0000, 0.0000, 0.0000, 0.0000, 4.1671, 0.0000,
    2.8295, 3.0353, 0.0000, 2.7780, 0.0000, 0.0000, 0.0000, 0.0000};
const int representative_64x16_weights_size = 64 * 16;
const int representative_64x16_weights_dims[] = {2, 16, 64};
const float representative_64x16_weights_data[] = {
    -0.1075, 0.1245,  0.1811,  -0.1302, -0.1868, 0.0679,  0.1245,  0.2321,
    -0.1981, -0.2094, 0.1358,  -0.1698, 0.0113,  0.0566,  0.1358,  -0.2490,
    0.0000,  -0.1189, -0.0170, -0.0396, -0.3113, 0.1641,  -0.4188, 0.0566,
    -0.4471, 0.4754,  -0.0396, 0.0113,  -0.0340, 0.0170,  0.0170,  0.1811,
    -0.0792, 0.4981,  0.2490,  -0.1924, 0.0792,  0.1868,  -0.1075, -0.3962,
    0.1358,  0.2547,  -0.1245, -0.0962, -0.0283, 0.4132,  -0.0057, -0.5150,
    0.1019,  0.1585,  -0.0962, -0.2207, -0.2377, 0.2830,  0.4471,  0.0170,
    0.0566,  0.2038,  0.1019,  -0.0226, 0.2830,  0.1415,  0.0283,  -0.0792,
    0.4301,  0.3226,  -0.1132, 0.4981,  -0.3849, -0.2943, -0.2547, -0.2264,
    0.0453,  -0.0170, 0.0396,  0.1415,  0.3000,  0.2547,  0.0962,  0.2151,
    -0.1585, -0.1302, -0.0057, -0.2773, 0.0283,  -0.0906, 0.1302,  -0.1075,
    -0.0566, 0.1755,  0.2773,  0.0283,  0.0566,  0.1528,  -0.0736, -0.2830,
    0.0792,  0.0962,  -0.2321, -0.0113, 0.2660,  -0.2887, -0.0566, 0.0057,
    -0.2547, -0.0679, -0.2321, 0.0340,  0.1868,  0.2490,  0.2264,  -0.3509,
    0.1585,  -0.0849, -0.0623, 0.1132,  0.3396,  -0.2490, 0.1528,  0.0679,
    0.1755,  0.4754,  -0.0057, -0.2151, -0.1415, -0.1302, -0.2717, 0.1641,
    0.5037,  -0.2321, 0.0170,  -0.1755, -0.1075, -0.0226, 0.2038,  -0.0340,
    -0.5150, -0.3113, 0.1472,  -0.0226, 0.1528,  0.1189,  -0.1472, 0.0396,
    -0.3000, -0.1924, -0.0283, 0.0283,  0.1641,  0.0736,  0.1472,  -0.1755,
    -0.1132, 0.0113,  -0.1868, -0.2604, -0.3283, -0.0509, 0.0283,  -0.0679,
    0.0623,  0.0792,  -0.0283, -0.0962, 0.0396,  0.1641,  0.4584,  0.3226,
    0.0226,  -0.1811, 0.2377,  -0.1019, 0.2321,  0.1811,  -0.1924, -0.0057,
    0.0736,  0.0113,  0.2547,  -0.2264, -0.0170, -0.0396, 0.1245,  -0.1415,
    0.1755,  0.3679,  -0.2377, -0.0396, -0.1585, -0.3000, -0.1641, -0.1302,
    -0.0396, -0.1698, 0.1189,  0.2434,  0.1132,  -0.1245, -0.1415, 0.0453,
    0.1868,  -0.0906, -0.1189, -0.0509, 0.0057,  -0.1189, -0.0057, 0.0170,
    -0.1924, 0.2207,  0.0792,  -0.4641, -0.2660, 0.2943,  0.1358,  -0.0340,
    -0.3339, -0.1189, 0.0906,  -0.4358, 0.0453,  -0.1755, 0.1415,  0.0340,
    0.1924,  -0.0057, 0.2321,  -0.2094, -0.1132, 0.0000,  0.1924,  -0.3000,
    0.0340,  -0.3396, -0.0906, -0.0340, 0.1641,  -0.0226, -0.1472, -0.1019,
    0.2377,  -0.0962, -0.3396, -0.5433, 0.0906,  0.2151,  -0.0679, 0.1755,
    0.1528,  0.0283,  -0.4188, -0.0340, -0.0057, -0.0679, 0.0509,  0.1472,
    -0.3849, -0.0113, 0.3962,  0.0849,  0.1472,  0.0340,  -0.1358, 0.1641,
    -0.2038, 0.2151,  -0.1189, -0.3679, 0.0906,  -0.0679, 0.5716,  -0.0057,
    -0.0736, 0.0113,  0.2830,  -0.2887, 0.0396,  0.0849,  -0.0736, -0.0736,
    -0.3679, 0.2264,  0.0113,  -0.1641, 0.0396,  -0.1132, -0.0623, 0.3113,
    0.5999,  -0.1415, 0.1472,  -0.2038, -0.1132, -0.2377, 0.0566,  0.1755,
    -0.0057, -0.0453, 0.0226,  0.1132,  0.1698,  0.0340,  -0.0226, 0.0226,
    0.4415,  -0.3792, 0.0792,  0.3736,  -0.5999, -0.3056, -0.1924, -0.1132,
    -0.0962, 0.0283,  0.0000,  -0.3339, -0.3226, 0.3679,  -0.0453, -0.1641,
    0.0170,  0.1302,  -0.0170, -0.0509, 0.1755,  -0.0283, -0.1302, -0.2887,
    -0.0679, 0.0340,  0.4641,  0.2321,  0.7188,  0.3339,  -0.1075, 0.4754,
    -0.0226, 0.3226,  -0.1528, -0.0849, 0.0509,  -0.1981, 0.0113,  0.2321,
    0.2773,  -0.1019, 0.4075,  0.0396,  0.0792,  0.1132,  -0.0906, -0.4188,
    0.1924,  -0.3679, -0.6396, 0.1358,  0.4981,  0.4132,  -0.0283, 0.3849,
    -0.3509, -0.0566, -0.0962, 0.3113,  -0.1811, 0.4019,  0.0453,  -0.0057,
    -0.1868, -0.2490, -0.0792, -0.3622, 0.1924,  -0.0453, -0.1528, -0.1811,
    0.5943,  -0.1302, 0.3170,  -0.0170, 0.0509,  -0.1528, -0.1755, 0.5547,
    0.2490,  -0.0906, 0.0000,  0.1698,  0.0000,  0.0340,  -0.1132, -0.0509,
    -0.1755, -0.2943, 0.1472,  0.0849,  0.0000,  0.1528,  -0.0566, 0.1528,
    -0.5264, -0.5320, -0.0736, 0.0566,  0.2604,  -0.4075, 0.0962,  -0.3453,
    -0.1415, 0.0057,  0.3905,  0.2830,  0.3679,  0.5320,  -0.2660, 0.0340,
    0.0736,  0.0057,  0.2207,  0.4471,  0.0849,  0.3000,  -0.0057, -0.0623,
    0.1415,  -0.0566, 0.5264,  -0.0340, 0.0226,  -0.0623, -0.0113, -0.5037,
    -0.4471, 0.0170,  -0.0396, -0.1358, -0.1698, 0.1924,  0.0057,  -0.1585,
    0.0849,  -0.1698, 0.0057,  -0.1245, -0.0170, -0.1755, -0.0792, 0.5264,
    0.1358,  0.2434,  0.1585,  -0.4188, -0.1472, -0.1358, -0.0849, -0.1189,
    0.5037,  0.0736,  -0.0453, -0.2434, 0.1868,  -0.0679, 0.1415,  -0.2717,
    0.2604,  0.0057,  -0.1528, -0.1811, 0.0226,  -0.1641, 0.3170,  -0.1981,
    0.1245,  0.0226,  0.0566,  0.2830,  -0.1755, 0.0396,  -0.2094, 0.1924,
    0.1698,  0.0283,  0.1641,  0.0849,  0.0000,  -0.1698, -0.1415, -0.3000,
    0.4471,  0.3056,  -0.0283, -0.4245, -0.0453, 0.0226,  0.0000,  -0.1075,
    -0.1528, -0.3226, 0.2773,  -0.2264, -0.1811, 0.1755,  -0.3566, -0.4188,
    0.1755,  -0.0057, 0.2038,  0.1075,  0.3679,  -0.0792, 0.2207,  -0.0453,
    0.3736,  0.2943,  -0.0113, -0.0623, 0.2264,  0.0113,  -0.0396, -0.2207,
    0.0453,  -0.2830, -0.1302, 0.0623,  -0.1924, -0.1811, -0.2717, 0.2830,
    0.2094,  0.0170,  -0.3170, -0.0283, -0.1189, -0.0509, -0.0566, -0.3622,
    0.1132,  -0.0906, 0.1132,  0.4019,  -0.4698, -0.1019, -0.1075, -0.2094,
    -0.2207, -0.0509, 0.0057,  0.1019,  -0.0509, 0.2264,  -0.5716, 0.0226,
    -0.4019, 0.1641,  -0.3000, 0.3849,  0.1245,  0.0679,  0.3056,  0.2377,
    0.0679,  -0.0170, -0.5377, -0.0170, 0.0057,  0.1358,  -0.1132, -0.2038,
    0.0679,  0.1075,  -0.2773, 0.5943,  0.0623,  -0.1472, 0.3566,  0.0396,
    -0.2377, 0.2604,  0.0849,  0.1358,  -0.3792, -0.0340, -0.1415, 0.3566,
    -0.3736, 0.1245,  0.0566,  0.3396,  0.0736,  0.4019,  -0.1528, 0.1075,
    0.0792,  -0.2547, 0.0453,  -0.1755, 0.1868,  -0.2547, 0.1075,  0.0623,
    0.1698,  -0.0170, 0.1585,  -0.0736, -0.4358, -0.0113, -0.6792, -0.0849,
    -0.0396, -0.6056, 0.1358,  0.1189,  0.2547,  0.1528,  0.2887,  0.0453,
    -0.1075, -0.3283, -0.0453, -0.0509, 0.2038,  0.2547,  0.0849,  -0.0566,
    -0.1698, 0.0509,  -0.0113, -0.1585, 0.1924,  -0.0792, -0.1868, 0.0509,
    -0.1698, -0.0849, -0.0170, 0.0453,  0.3170,  0.0906,  -0.5943, -0.1245,
    0.1585,  -0.1755, -0.2151, 0.0906,  0.1924,  0.3170,  -0.2490, -0.5660,
    -0.0283, 0.0962,  -0.1358, 0.1585,  0.0057,  -0.2604, 0.1189,  -0.0170,
    0.3509,  0.0623,  0.0679,  -0.1302, -0.0792, 0.0906,  -0.0792, 0.0849,
    -0.1924, 0.2604,  -0.1245, -0.3679, 0.0340,  0.0113,  -0.1698, 0.2490,
    0.0283,  0.1019,  -0.3736, 0.1019,  -0.2207, -0.0340, 0.3170,  0.1755,
    0.0962,  0.3226,  -0.0113, -0.1189, -0.2321, -0.0226, -0.2434, -0.0170,
    -0.1585, -0.0283, -0.1132, 0.0679,  -0.4188, -0.0453, 0.1528,  -0.1302,
    -0.3792, 0.1415,  -0.1358, -0.1811, 0.1302,  0.1415,  0.5207,  0.0509,
    -0.1358, -0.0396, -0.2434, 0.0396,  0.0792,  -0.2264, -0.1415, 0.0906,
    0.1245,  0.0170,  0.0623,  -0.1415, 0.2773,  -0.3566, -0.0396, 0.2887,
    0.4188,  0.1698,  -0.2547, 0.1132,  -0.0453, -0.0113, -0.1358, 0.1075,
    0.0566,  0.1075,  0.2604,  -0.0849, -0.2490, 0.1415,  0.0509,  -0.2151,
    0.0340,  0.1698,  0.0509,  -0.0906, 0.0566,  -0.1075, -0.2151, 0.2038,
    -0.1924, -0.0113, 0.2830,  0.1358,  -0.1189, 0.0113,  -0.5603, -0.2830,
    -0.2943, 0.0453,  -0.0396, 0.1358,  0.0566,  0.2038,  -0.3283, -0.0509,
    0.0509,  0.1641,  0.2094,  -0.2038, -0.1868, -0.1585, -0.2207, -0.1302,
    0.0396,  -0.1019, -0.0679, 0.1075,  -0.4584, -0.2207, 0.2434,  -0.0113,
    0.0849,  0.1755,  -0.3056, 0.1585,  -0.2547, 0.0453,  0.0906,  -0.1358,
    -0.0679, -0.0509, 0.0679,  -0.3509, 0.0057,  0.0453,  0.4132,  -0.1981,
    0.2264,  -0.0736, 0.1075,  0.0679,  -0.0906, -0.3113, 0.0509,  0.0849,
    0.2604,  0.0623,  -0.3113, 0.3849,  0.0000,  0.6396,  -0.2038, -0.1019,
    0.1245,  -0.0453, 0.1641,  0.1075,  -0.1075, -0.2660, -0.4528, -0.0566,
    -0.0170, 0.0453,  0.0340,  0.1189,  -0.2434, -0.0283, -0.1811, 0.2547,
    0.0000,  -0.0226, 0.4471,  0.1019,  -0.1472, 0.0849,  0.1075,  0.1075,
    0.0283,  -0.2773, 0.4415,  -0.1811, 0.2717,  0.3170,  0.0509,  0.0623,
    -0.0962, 0.1585,  -0.0792, -0.1811, -0.0792, -0.3283, 0.0962,  -0.1698,
    -0.0736, 0.0453,  0.0962,  -0.3566, -0.4584, 0.3396,  -0.4811, 0.3056,
    -0.1755, 0.2490,  -0.1698, -0.2377, -0.3339, -0.0453, 0.1811,  0.0736,
    0.0340,  -0.0962, -0.0113, -0.3056, -0.3339, 0.2038,  0.2038,  -0.1924,
    0.2547,  -0.4471, -0.0849, -0.2038, 0.3566,  -0.4811, 0.3453,  0.0849,
    0.1189,  0.3170,  -0.1358, 0.2717,  0.0113,  -0.4754, -0.1924, 0.4245,
    -0.2773, 0.3453,  0.2264,  0.2943,  0.5320,  0.2773,  -0.2264, -0.1019,
    -0.1132, -0.3962, 0.3679,  0.0509,  -0.0623, -0.0906, -0.5603, -0.1641,
    -0.3170, -0.2377, 0.1415,  -0.0509, 0.0792,  0.0170,  -0.0226, -0.0057,
    -0.1358, -0.4245, 0.3905,  0.3113,  0.0340,  -0.1189, 0.2887,  -0.2943,
    -0.3056, 0.2434,  0.1019,  -0.0170, 0.3849,  0.1528,  -0.0736, -0.0170,
    0.0792,  0.1755,  0.0509,  0.3509,  0.1472,  0.1528,  0.1472,  0.0057,
    0.0113,  -0.0113, -0.3283, -0.3962, -0.0792, -0.1245, -0.0283, -0.1868,
    0.4019,  0.2943,  -0.0906, -0.2321, 0.6056,  0.1189,  0.0340,  -0.2207,
    -0.0453, 0.3339,  0.2377,  -0.1641, 0.3736,  0.2151,  -0.2547, 0.0453,
    0.1924,  -0.1019, -0.0340, -0.2207, 0.3962,  -0.4471, -0.2547, -0.2151,
    -0.3736, 0.0283,  0.1189,  0.0283,  0.0736,  0.0396,  0.1019,  0.0283,
    0.0170,  0.2321,  0.3509,  -0.0226, -0.0226, 0.0736,  0.0283,  0.1641,
    -0.0906, 0.1811,  0.0226,  0.5716,  -0.0396, -0.0509, -0.1641, -0.0509,
    0.4132,  -0.2604, 0.1019,  -0.0283, -0.0340, 0.0453,  0.1472,  -0.0057,
    0.2717,  -0.2094, 0.3396,  0.0340,  0.1245,  0.2547,  -0.5886, 0.2717,
    -0.0906, 0.1641,  0.0962,  -0.0792, -0.0113, 0.2264,  -0.0736, 0.3170,
    0.0623,  0.0679,  0.0623,  -0.0792, -0.2207, 0.1924,  0.1245,  -0.2773};
const int representative_64x16_bias_dims[] = {1, 16};
const float representative_64x16_bias_data[] = {
    -0.0084, 0.0006,  0.0000,  0.0000,  -0.0087, -0.0006, -0.0003, -0.0003,
    0.0006,  -0.0003, -0.0003, -0.0003, -0.0253, 0.0012,  0.0000,  0.0000};
const float representative_64x16_golden[] = {
    3.8624,  -2.9580, 4.3043,  -1.2844, -1.5769, -2.7998, -0.1011, -3.4029,
    -1.0557, -7.1931, -1.4852, -0.4163, 1.7186,  -0.6965, 0.3580,  2.7378};
const int representative_64x16_output_size = 16;
const int representative_64x16_output_dims[] = {2, 1, 16};

template <typename T>
TfLiteStatus ValidateFullyConnectedGoldens(
    TfLiteTensor* tensors, const int tensors_size,
    const TfLiteFusedActivation activation, const float tolerance,
    const int output_len, const T* golden, T* output_data) {
  TfLiteFullyConnectedParams builtin_data = {
      activation, kTfLiteFullyConnectedWeightsFormatDefault, false, false};

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_FULLY_CONNECTED();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TfLiteStatus status = runner.InitAndPrepare();
  if (status != kTfLiteOk) {
    return status;
  }

  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
  return kTfLiteOk;
}

#if !defined(XTENSA)  // Needed to avoid build error from unused functions.
TfLiteStatus TestFullyConnectedFloat(
    const int* input_dims_data, const float* input_data,
    const int* weights_dims_data, const float* weights_data,
    const int* bias_dims_data, const float* bias_data, const float* golden,
    const int* output_dims_data, TfLiteFusedActivation activation,
    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(weights_data, weights_dims),
      CreateTensor(bias_data, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  return ValidateFullyConnectedGoldens(tensors, tensors_size, activation, 1e-4f,
                                       output_dims_count, golden, output_data);
}
#endif

template <typename T>
TfLiteStatus TestFullyConnectedQuantized(
    const int* input_dims_data, const float* input_data, T* input_quantized,
    const float input_scale, const int input_zero_point,
    const int* weights_dims_data, const float* weights_data,
    T* weights_quantized, const float weights_scale,
    const int weights_zero_point, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_quantized, const float* golden,
    T* golden_quantized, const int* output_dims_data, const float output_scale,
    const int output_zero_point, TfLiteFusedActivation activation,
    T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(weights_data, weights_quantized, weights_dims,
                            weights_scale, weights_zero_point),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, weights_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);

  return ValidateFullyConnectedGoldens(tensors, tensors_size, activation, 0.0f,
                                       output_dims_count, golden_quantized,
                                       output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

#if !defined(XTENSA) && !defined(CEVA_BX1) && !defined(CEVA_SP500)
// TODO(b/170503075): xtensa kernels are less general
// than reference kernels and we ifdef out test cases that are currently known
// to fail.

// CEVA's fully connected implementation assumes weights_zero_point=0 as
// described in TFLite's quantization specification. tests which use a different
// zero point will so ifdefed out.
// See tflite quantization spec:
// https://www.tensorflow.org/lite/performance/quantization_spec
TF_LITE_MICRO_TEST(SimpleTest) {
  float output_data[tflite::testing::simple_output_size];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 128;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  uint8_t input_quantized[tflite::testing::simple_input_size];
  uint8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  uint8_t golden_quantized[tflite::testing::simple_output_size];
  uint8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}
#endif

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;

  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  const int input_dims_4d[] = {4, 1, 1, 2, 10};

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          input_dims_4d, tflite::testing::simple_input_data, input_quantized,
          input_scale, input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8Relu) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 0;

  const float output_scale = 0.5f;
  const int output_zero_point = -128;

  int8_t input_quantized[tflite::testing::relu_input_size];
  int8_t weights_quantized[tflite::testing::relu_weights_size];
  int32_t bias_quantized[tflite::testing::relu_output_size];
  int8_t golden_quantized[tflite::testing::relu_output_size];
  int8_t output_data[tflite::testing::relu_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::relu_input_dims, tflite::testing::relu_input_data,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::relu_weights_dims,
          tflite::testing::relu_weights_data, weights_quantized, weights_scale,
          weights_zero_point, tflite::testing::relu_bias_dims,
          tflite::testing::relu_bias_data, bias_quantized,
          tflite::testing::relu_golden, golden_quantized,
          tflite::testing::relu_output_dims, output_scale, output_zero_point,
          kTfLiteActRelu, output_data),
      kTfLiteOk);
}

#if !defined(XTENSA)  // TODO(b/170503075): xtensa kernels are less general than
                      // reference kernels and we ifdef out test cases that are
                      // currently known to fail.
TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8Relu) {
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 128;

  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  uint8_t input_quantized[tflite::testing::relu_input_size];
  uint8_t weights_quantized[tflite::testing::relu_weights_size];
  int32_t bias_quantized[tflite::testing::relu_output_size];
  uint8_t golden_quantized[tflite::testing::relu_output_size];
  uint8_t output_data[tflite::testing::relu_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::relu_input_dims, tflite::testing::relu_input_data,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::relu_weights_dims,
          tflite::testing::relu_weights_data, weights_quantized, weights_scale,
          weights_zero_point, tflite::testing::relu_bias_dims,
          tflite::testing::relu_bias_data, bias_quantized,
          tflite::testing::relu_golden, golden_quantized,
          tflite::testing::relu_output_dims, output_scale, output_zero_point,
          kTfLiteActRelu, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInput) {
  const int input_dims_4d[] = {4, 1, 1, 2, 10};

  float output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_4d, tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedUInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const float weights_scale = 1.0f;
  const int weights_zero_point = 128;

  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  const int input_dims_4d[] = {4, 1, 1, 2, 10};

  uint8_t input_quantized[tflite::testing::simple_input_size];
  uint8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  uint8_t golden_quantized[tflite::testing::simple_output_size];
  uint8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          input_dims_4d, tflite::testing::simple_input_data, input_quantized,
          input_scale, input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, weights_zero_point, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(Representative1x64Input1x16Output) {
  float output_data[tflite::testing::representative_64x16_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::representative_64x16_input_dims,
          tflite::testing::representative_64x16_input_data,
          tflite::testing::representative_64x16_weights_dims,
          tflite::testing::representative_64x16_weights_data,
          tflite::testing::representative_64x16_bias_dims,
          tflite::testing::representative_64x16_bias_data,
          tflite::testing::representative_64x16_golden,
          tflite::testing::representative_64x16_output_dims, kTfLiteActNone,
          output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(Representative1x64Input1x16OutputQuantizedUInt8) {
  const float input_scale = 0.051445;
  const int input_zero_point = 0;
  const float weights_scale = 0.005660;
  const int weights_zero_point = 128;

  const float output_scale = 0.069785;
  const int output_zero_point = 119;

  uint8_t input_quantized[tflite::testing::representative_64x16_input_size];
  uint8_t weights_quantized[tflite::testing::representative_64x16_weights_size];
  int32_t bias_quantized[tflite::testing::representative_64x16_output_size];
  uint8_t golden_quantized[tflite::testing::representative_64x16_output_size];
  uint8_t output_data[tflite::testing::representative_64x16_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::representative_64x16_input_dims,
          tflite::testing::representative_64x16_input_data, input_quantized,
          input_scale, input_zero_point,
          tflite::testing::representative_64x16_weights_dims,
          tflite::testing::representative_64x16_weights_data, weights_quantized,
          weights_scale, weights_zero_point,
          tflite::testing::representative_64x16_bias_dims,
          tflite::testing::representative_64x16_bias_data, bias_quantized,
          tflite::testing::representative_64x16_golden, golden_quantized,
          tflite::testing::representative_64x16_output_dims, output_scale,
          output_zero_point, kTfLiteActNone, output_data),
      kTfLiteOk);
}

#endif

TF_LITE_MICRO_TEST(Representative1x64Input1x16OutputQuantizedInt8) {
  const float input_scale = 0.051445;
  const int input_zero_point = -128;
  const float weights_scale = 0.005660;
  const int weights_zero_point = 0;

  const float output_scale = 0.069785;
  const int output_zero_point = -9;

  int8_t input_quantized[tflite::testing::representative_64x16_input_size];
  int8_t weights_quantized[tflite::testing::representative_64x16_weights_size];
  int32_t bias_quantized[tflite::testing::representative_64x16_output_size];
  int8_t golden_quantized[tflite::testing::representative_64x16_output_size];
  int8_t output_data[tflite::testing::representative_64x16_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::representative_64x16_input_dims,
          tflite::testing::representative_64x16_input_data, input_quantized,
          input_scale, input_zero_point,
          tflite::testing::representative_64x16_weights_dims,
          tflite::testing::representative_64x16_weights_data, weights_quantized,
          weights_scale, weights_zero_point,
          tflite::testing::representative_64x16_bias_dims,
          tflite::testing::representative_64x16_bias_data, bias_quantized,
          tflite::testing::representative_64x16_golden, golden_quantized,
          tflite::testing::representative_64x16_output_dims, output_scale,
          output_zero_point, kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TESTS_END
