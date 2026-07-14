/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_FLIP_LEFT_RIGHT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_FLIP_LEFT_RIGHT_H_

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace flip_left_right {

// Flip (left_right)
//
// Inputs: [img: any]
// Ouputs: [img: any]
//
// Flips the given image horizontally (left to right).
// Mimics semantic of `tf.image.flip_left_right`.

// https://www.tensorflow.org/api_docs/python/tf/image/flip_left_right

const algo::Algo* Impl_FlipLeftRight();

}  // namespace flip_left_right
}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_FLIP_LEFT_RIGHT_H_
