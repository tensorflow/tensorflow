/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

#include <cmath>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {
// Matrices for Winograd trasformations were computed with the method described
// here https://openreview.net/pdf?id=H1ZaRZVKg
std::vector<float> GetTransposedMatrixForWinograd(int width, int height) {
  const float kDelta = std::sqrt(2.0f) / 2.0f;
  std::vector<float> px(width);

  px[0] = 0.0f;
  const int points_count = (width - 1) / 2;
  for (int i = 0; i < points_count; ++i) {
    px[i * 2 + 1] = kDelta * (i + 1.0f);
    px[i * 2 + 2] = -kDelta * (i + 1.0f);
  }
  px[width - 1] = 1.0f;

  std::vector<float> py(width, 1.0f);
  py[width - 1] = 0.0f;

  std::vector<float> result(height * width);
  for (int y = 0; y < width; ++y) {
    for (int x = 0; x < height; ++x) {
      result[x * width + y] =
          std::pow(px[y], 1.0f * x) * std::pow(py[y], (height - 1.0f) - x);
    }
  }
  return result;
}

std::vector<float> GetInversedMatrixForWinograd(int rank) {
  auto matrix = GetTransposedMatrixForWinograd(rank, rank);
  std::vector<float> inverted(rank * rank, 0.0f);
  for (int i = 0; i < rank; ++i) {
    inverted[i * rank + i] = 1.0f;
  }

  for (int i = 1; i < rank - 1; ++i) {
    float inv_t = 1.0f / matrix[i * rank + i];
    for (int x = i; x < rank; ++x) {
      matrix[i * rank + x] *= inv_t;
    }
    for (int x = 0; x < rank; ++x) {
      inverted[i * rank + x] *= inv_t;
    }

    for (int y = 0; y < rank; ++y) {
      if (y == i) continue;
      float t = matrix[y * rank + i];
      for (int x = i; x < rank; ++x) {
        matrix[y * rank + x] -= t * matrix[i * rank + x];
      }
      for (int x = 0; x < rank; ++x) {
        inverted[y * rank + x] -= t * inverted[i * rank + x];
      }
    }
  }

  return inverted;
}

std::vector<float> Multiply(const std::vector<float>& a_mat,
                            const std::vector<float>& b_mat, int m, int n,
                            int k) {
  std::vector<float> result(m * k);
  for (int y = 0; y < m; ++y) {
    for (int x = 0; x < k; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < n; ++i) {
        sum += a_mat[y * n + i] * b_mat[i * k + x];
      }
      result[y * k + x] = sum;
    }
  }
  return result;
}
}  // namespace

std::vector<float> AtMatrixForWinograd4x4To6x6() {
  return GetTransposedMatrixForWinograd(6, 4);
}

std::vector<float> BtMatrixForWinograd4x4To6x6() {
  return GetInversedMatrixForWinograd(6);
}

void RearrangeWeightsToWinograd4x4To6x6Weights(
    const Tensor<OHWI, DataType::FLOAT32>& src_weights,
    Tensor<OHWI, DataType::FLOAT32>* dst_weights) {
  OHWI dst_shape;
  dst_shape.o = src_weights.shape.o;
  dst_shape.h = 6;
  dst_shape.w = 6;
  dst_shape.i = src_weights.shape.i;
  dst_weights->shape = dst_shape;
  dst_weights->data.resize(dst_shape.DimensionsProduct());

  auto gt_mat = GetTransposedMatrixForWinograd(6, 3);
  std::vector<float> g_mat(gt_mat.size());
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 6; ++x) {
      g_mat[x * 3 + y] = gt_mat[y * 6 + x];
    }
  }

  for (int d = 0; d < src_weights.shape.o; ++d) {
    for (int s = 0; s < src_weights.shape.i; ++s) {
      std::vector<float> in_vals(9);
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
          const int f_index = src_weights.shape.LinearIndex({d, y, x, s});
          in_vals[y * 3 + x] = src_weights.data[f_index];
        }
      }

      auto temp_vals = Multiply(g_mat, in_vals, 6, 3, 3);
      auto out_vals = Multiply(temp_vals, gt_mat, 6, 3, 6);
      for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
          const int f_index = dst_shape.LinearIndex({d, y, x, s});
          dst_weights->data[f_index] = out_vals[y * 6 + x];
        }
      }
    }
  }
}

bool IsSuitableForWinograd4x4To6x6(const Convolution2DAttributes& attr) {
  return attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
         attr.dilations == HW(1, 1) && attr.strides == HW(1, 1) &&
         attr.groups == 1;
}

}  // namespace gpu
}  // namespace tflite
