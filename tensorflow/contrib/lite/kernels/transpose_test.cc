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
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"

namespace tflite {
namespace {

void RunTestPermutation(const std::vector<int>& shape,
                        const std::vector<int>& perms,
                        std::vector<float>* input_transposed) {
  // Count elements and allocate output.
  int count = 1;
  for (auto factor : shape) count *= factor;
  input_transposed->resize(count);

  // Create the dummy data
  std::vector<float> input(count);
  for (int i = 0; i < input.size(); i++) {
    input[i] = i;
  }

  // Create reversed and padded perms.
  int reversed_perms[4];
  for (int output_k = 0, input_k = shape.size() - 1; output_k < shape.size();
       output_k++, input_k--) {
    reversed_perms[output_k] = shape.size() - perms[input_k] - 1;
  }
  // Unused dimensions should not be permuted so pad with identity transform
  // subset.
  for (int k = shape.size(); k < 4; k++) {
    reversed_perms[k] = k;
  }

  // Make input and output dims (i.e. reversed shape and dest_shape).
  Dims<4> input_dims = GetTensorDims(shape);
  Dims<4> output_dims;
  for (int i = 0; i < 4; i++) {
    output_dims.sizes[i] = input_dims.sizes[reversed_perms[i]];
  }
  output_dims.strides[0] = 1;
  for (int k = 1; k < 4; k++) {
    output_dims.strides[k] =
        output_dims.strides[k - 1] * output_dims.sizes[k - 1];
  }

  reference_ops::Transpose<float>(input.data(), input_dims,
                                  input_transposed->data(), output_dims,
                                  reversed_perms);
}

TEST(TransposeTest, Test1D) {
  // Basic 1D identity.
  std::vector<float> out;
  RunTestPermutation({3}, {0}, &out);
  ASSERT_EQ(out, std::vector<float>({0, 1, 2}));
}

TEST(TransposeTest, Test2D) {
  std::vector<float> out;
  // Basic 2D.
  RunTestPermutation({3, 2}, {1, 0}, &out);
  ASSERT_EQ(out, std::vector<float>({0, 2, 4, 1, 3, 5}));
  // Identity.
  RunTestPermutation({3, 2}, {0, 1}, &out);
  ASSERT_EQ(out, std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST(TransposeTest, Test3D) {
  std::vector<float> out;
  // Test 3 dimensional
  {
    std::vector<float> ref({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                            2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});
    RunTestPermutation({2, 3, 4}, {2, 0, 1}, &out);
    ASSERT_EQ(out, ref);
  }
  // Test 3 dimensional identity transform
  {
    RunTestPermutation({2, 3, 4}, {0, 1, 2}, &out);
    std::vector<float> ref(out.size());
    for (int k = 0; k < ref.size(); k++) ref[k] = k;
    ASSERT_EQ(out, ref);
  }
}

TEST(TransposeTest, Test4D) {
  std::vector<float> out;
  // Basic 4d.
  RunTestPermutation({2, 3, 4, 5}, {2, 0, 1, 3}, &out);
  ASSERT_EQ(
      out,
      std::vector<float>(
          {0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
           60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
           5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
           65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
           10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
           70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
           15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
           75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119}));
  RunTestPermutation({2, 3, 4, 5}, {0, 1, 2, 3}, &out);
  // Basic identity.
  std::vector<float> ref(out.size());
  for (int k = 0; k < ref.size(); k++) ref[k] = k;
  ASSERT_EQ(out, ref);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
