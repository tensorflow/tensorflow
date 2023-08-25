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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_RESIZE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_RESIZE_H_

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"

namespace ml_adj {
namespace resize {

// Bilinear Resize
//
// Inputs: [img: float, new size: vector<unsigned>]
// Ouputs: [img: float]
//
// Resizes the given image, scaling height and width dimensions to `new size`.
// Mimics semantic of `tf.image.resize`, where `method` equals to bilinear
// interpolation.
//
// https://www.tensorflow.org/api_docs/python/tf/image/resize

const algo::Algo* Impl_Resize();

}  // namespace resize
}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_RESIZE_H_
