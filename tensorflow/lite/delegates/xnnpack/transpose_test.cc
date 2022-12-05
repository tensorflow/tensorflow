/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/transpose_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(Transpose, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0};
  // clang-format off
  TransposeTester()
      .num_dims(1)
      .input_shape({37})
      .perm(perm)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
  // clang-format on
}

TEST(Transpose, 2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1};
  do {
    // clang-format off
    TransposeTester()
        .num_dims(2)
        .input_shape({37, 113})
        .perm(perm)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(Transpose, 3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1, 2};
  do {
    TransposeTester()
        .num_dims(3)
        .input_shape({5, 7, 11})
        .perm(perm)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(Transpose, 4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1, 2, 3};
  do {
    TransposeTester()
        .num_dims(4)
        .input_shape({5, 7, 11, 13})
        .perm(perm)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(Transpose, 5D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeTester()
        .num_dims(5)
        .input_shape({3, 5, 7, 11, 13})
        .perm(perm)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
  } while (std::next_permutation(perm.begin(), perm.end()));
}

}  // namespace xnnpack
}  // namespace tflite
