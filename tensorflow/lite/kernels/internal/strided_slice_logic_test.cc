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
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"

#include <gtest/gtest.h>

namespace tflite {
namespace {

void RunStridedSlicePadIndices(std::initializer_list<int> begin,
                               std::initializer_list<int> end,
                               std::initializer_list<int> stride,
                               std::initializer_list<int> expected_begin,
                               std::initializer_list<int> expected_end,
                               std::initializer_list<int> expected_stride) {
  StridedSliceParams op_params;
  int dims = begin.size();
  op_params.start_indices_count = dims;
  op_params.stop_indices_count = dims;
  op_params.strides_count = dims;

  for (int i = 0; i < dims; ++i) {
    op_params.start_indices[i] = begin.begin()[i];
    op_params.stop_indices[i] = end.begin()[i];
    op_params.strides[i] = stride.begin()[i];
  }

  strided_slice::StridedSlicePadIndices(&op_params, 4);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(op_params.start_indices[i], expected_begin.begin()[i]);
    EXPECT_EQ(op_params.stop_indices[i], expected_end.begin()[i]);
    EXPECT_EQ(op_params.strides[i], expected_stride.begin()[i]);
  }
}

TEST(RunStridedSlicePadIndices, Pad1) {
  RunStridedSlicePadIndices({1, 2, 3},     // begin
                            {4, 5, 6},     // end
                            {2, 2, 2},     // stride
                            {0, 1, 2, 3},  // expected_begin
                            {1, 4, 5, 6},  // expected_end
                            {1, 2, 2, 2}   // expected_stride
  );
}

TEST(RunStridedSlicePadIndices, Pad2) {
  RunStridedSlicePadIndices({1, 2},        // begin
                            {4, 5},        // end
                            {2, 2},        // stride
                            {0, 0, 1, 2},  // expected_begin
                            {1, 1, 4, 5},  // expected_end
                            {1, 1, 2, 2}   // expected_stride
  );
}

TEST(RunStridedSlicePadIndices, Pad3) {
  RunStridedSlicePadIndices({1},           // begin
                            {4},           // end
                            {2},           // stride
                            {0, 0, 0, 1},  // expected_begin
                            {1, 1, 1, 4},  // expected_end
                            {1, 1, 1, 2}   // expected_stride
  );
}

}  // namespace
}  // namespace tflite
