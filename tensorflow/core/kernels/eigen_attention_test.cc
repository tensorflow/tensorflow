/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/eigen_attention.h"
#include "tensorflow/core/platform/test.h"

namespace Eigen {

namespace {
void EigenApprox(float a, float b) {
  ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3);
}
}  // namespace

TEST(EigenAttentionTest, Simple) {
  const ptrdiff_t depth = 3;
  const ptrdiff_t batch = 10;
  const ptrdiff_t rows = 32;
  const ptrdiff_t cols = 48;
  const ptrdiff_t glimpse_rows = 8;
  const ptrdiff_t glimpse_cols = 6;

  Tensor<float, 4> input(depth, rows, cols, batch);
  input.setRandom();

  std::vector<IndexPair<float>> offsets;
  offsets.resize(batch);
  for (int i = 0; i < batch; ++i) {
    offsets[i].first = (-5 + i) / 10.0f;
    offsets[i].second = (5 - i) / 10.0f;
  }

  Tensor<float, 4> result(depth, glimpse_rows, glimpse_cols, batch);
  result = ExtractGlimpses(input, glimpse_rows, glimpse_cols, offsets);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < glimpse_cols; ++c) {
      ptrdiff_t source_c =
          c + ((1.0f + offsets[b].second) * cols - glimpse_cols) / 2;
      for (int r = 0; r < glimpse_rows; ++r) {
        ptrdiff_t source_r =
            r + ((1.0f + offsets[b].first) * rows - glimpse_rows) / 2;
        for (int d = 0; d < depth; ++d) {
          EigenApprox(result(d, r, c, b), input(d, source_r, source_c, b));
        }
      }
    }
  }
}

TEST(EigenAttentionTest, OutOfBoundsGlimpse) {
  const ptrdiff_t depth = 3;
  const ptrdiff_t batch = 10;
  const ptrdiff_t rows = 32;
  const ptrdiff_t cols = 48;
  const ptrdiff_t glimpse_rows = 8;
  const ptrdiff_t glimpse_cols = 6;

  Tensor<float, 4> input(depth, rows, cols, batch);
  input.setRandom();

  std::vector<IndexPair<float>> offsets;
  offsets.resize(batch);
  for (int i = 0; i < batch; ++i) {
    offsets[i].first = (-5 + i) / 2.0f;
    offsets[i].second = (5 - i) / 2.0f;
  }

  Tensor<float, 4> result(depth, glimpse_rows, glimpse_cols, batch);
  result = ExtractGlimpses(input, glimpse_rows, glimpse_cols, offsets);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < glimpse_cols; ++c) {
      ptrdiff_t source_c =
          c + ((1.0f + offsets[b].second) * cols - glimpse_cols) / 2;
      if (source_c < glimpse_cols / 2 || source_c >= cols - glimpse_cols / 2) {
        continue;
      }
      for (int r = 0; r < glimpse_rows; ++r) {
        ptrdiff_t source_r =
            r + ((1.0f + offsets[b].first) * rows - glimpse_rows) / 2;
        if (source_r < glimpse_rows / 2 ||
            source_r >= rows - glimpse_rows / 2) {
          continue;
        }
        for (int d = 0; d < depth; ++d) {
          EigenApprox(result(d, r, c, b), input(d, source_r, source_c, b));
        }
      }
    }
  }
}

}  // namespace Eigen
