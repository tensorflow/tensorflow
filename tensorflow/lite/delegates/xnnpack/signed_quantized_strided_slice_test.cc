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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/strided_slice_tester.h"

namespace tflite {
namespace xnnpack {

TEST_F(SignedQuantizedStridedSliceTest, 1D) {
  const std::vector<int32_t> input_shape = {RandomShape()};

  StridedSliceTester()
      .InputShape(input_shape)
      .RandomBegins(rng_)
      .RandomEnds(rng_)
      .Test(TensorType_INT8, xnnpack_delegate_.get());
}

TEST_F(SignedQuantizedStridedSliceTest, 2D) {
  const std::vector<int32_t> input_shape = {RandomShape(), RandomShape()};

  StridedSliceTester()
      .InputShape(input_shape)
      .RandomBegins(rng_)
      .RandomEnds(rng_)
      .Test(TensorType_FLOAT32, xnnpack_delegate_.get());
}

TEST_F(SignedQuantizedStridedSliceTest, 3D) {
  const std::vector<int32_t> input_shape = {RandomShape(), RandomShape(),
                                            RandomShape()};

  StridedSliceTester()
      .InputShape(input_shape)
      .RandomBegins(rng_)
      .RandomEnds(rng_)
      .Test(TensorType_FLOAT32, xnnpack_delegate_.get());
}

TEST_F(SignedQuantizedStridedSliceTest, 4D) {
  const std::vector<int32_t> input_shape = {RandomShape(), RandomShape(),
                                            RandomShape(), RandomShape()};

  StridedSliceTester()
      .InputShape(input_shape)
      .RandomBegins(rng_)
      .RandomEnds(rng_)
      .Test(TensorType_FLOAT32, xnnpack_delegate_.get());
}

TEST_F(SignedQuantizedStridedSliceTest, 5D) {
  const std::vector<int32_t> input_shape = {RandomShape(), RandomShape(),
                                            RandomShape(), RandomShape(),
                                            RandomShape()};

  StridedSliceTester()
      .InputShape(input_shape)
      .RandomBegins(rng_)
      .RandomEnds(rng_)
      .Test(TensorType_FLOAT32, xnnpack_delegate_.get());
}

}  // namespace xnnpack
}  // namespace tflite
