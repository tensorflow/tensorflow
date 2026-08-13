/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/select.h"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace reference_ops {
namespace {

// Regression test for a stack buffer overflow in BroadcastSelectSimple:
// cond_strides/x_strides/y_strides/o_strides/o_shape were fixed-size
// kMaxRank=8 C arrays, guarded only by a TFLITE_DCHECK_LE that compiles out
// in release builds. A rank-9 input -- reachable from a .tflite model's own
// declared tensor shapes for a Select/SelectV2 op -- wrote one element past
// each array. Same bug class and same fix (absl::InlinedVector, sized to
// the real rank) as broadcast_loop.h's rank-9 regression test.
TEST(SelectTest, SupportsRankGreaterThanInlineRank) {
  // x alternates 1/2 across all 9 dimensions; cond/y/output are constant 2.
  // This broadcasting pattern prevents BroadcastSelectSimple's loop-fusion
  // optimization from collapsing the dimensions, forcing next_dim_idx to
  // increment on (almost) every one of the 9 dimensions -- walking one past
  // the previously fixed-size 8-element arrays.
  const RuntimeShape cond_shape({2, 2, 2, 2, 2, 2, 2, 2, 2});
  const RuntimeShape x_shape({1, 2, 1, 2, 1, 2, 1, 2, 1});
  const RuntimeShape y_shape({2, 2, 2, 2, 2, 2, 2, 2, 2});
  const RuntimeShape output_shape({2, 2, 2, 2, 2, 2, 2, 2, 2});

  const size_t cond_size = cond_shape.FlatSize();
  const size_t x_size = x_shape.FlatSize();
  const size_t y_size = y_shape.FlatSize();
  const size_t output_size = output_shape.FlatSize();

  std::vector<uint8_t> cond_data(cond_size, 1);
  std::vector<float> x_data(x_size);
  std::vector<float> y_data(y_size, 100.0f);
  std::vector<float> output_data(output_size, 0.0f);
  for (size_t i = 0; i < x_size; ++i) x_data[i] = static_cast<float>(i);

  // Must not read or write outside the five metadata arrays regardless of
  // rank; ASan/stack-protector builds catch a regression here.
  BroadcastSelectSimple(cond_shape, cond_data.data(), x_shape, x_data.data(),
                        y_shape, y_data.data(), output_shape,
                        output_data.data());

  // Independently recompute the expected value for each output element via
  // a manual per-dimension index decomposition (not reusing any of
  // BroadcastSelectSimple's own stride/fusion logic), the same style of
  // correctness oracle broadcast_loop_test.cc uses.
  const int rank = output_shape.DimensionsCount();
  for (size_t output_index = 0; output_index < output_size; ++output_index) {
    size_t remaining = output_index;
    size_t cond_index = 0, x_index = 0, y_index = 0;
    size_t cond_stride = 1, x_stride = 1, y_stride = 1;
    // Decode output_index into per-dimension coordinates from the innermost
    // dimension outward, accumulating each input's flat index only where
    // that input's own dimension size is not 1 (i.e. not broadcasting).
    std::vector<size_t> coord(rank);
    for (int dim = rank - 1; dim >= 0; --dim) {
      const size_t output_dim = static_cast<size_t>(output_shape.Dims(dim));
      coord[dim] = remaining % output_dim;
      remaining /= output_dim;
    }
    for (int dim = rank - 1; dim >= 0; --dim) {
      const size_t cond_dim = static_cast<size_t>(cond_shape.Dims(dim));
      const size_t x_dim = static_cast<size_t>(x_shape.Dims(dim));
      const size_t y_dim = static_cast<size_t>(y_shape.Dims(dim));
      if (cond_dim != 1) cond_index += coord[dim] * cond_stride;
      if (x_dim != 1) x_index += coord[dim] * x_stride;
      if (y_dim != 1) y_index += coord[dim] * y_stride;
      cond_stride *= cond_dim;
      x_stride *= x_dim;
      y_stride *= y_dim;
    }
    const float expected = cond_data[cond_index] ? x_data[x_index] : y_data[y_index];
    EXPECT_EQ(output_data[output_index], expected)
        << "output index " << output_index;
  }
}

}  // namespace
}  // namespace reference_ops
}  // namespace tflite
