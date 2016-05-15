/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHTOSPACE_OP_H_
#define TENSORFLOW_CORE_KERNELS_BATCHTOSPACE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

// Functor used by BatchToSpaceOp to do the computations.
template <typename Device, typename T>
struct BatchToSpaceOpFunctor {
  // Implements the batch to space conversion.
  //
  // input: 4-D input tensor.
  // crops: [2, 2] matrix, [[crop_top, crop_bottom], [crop_left, crop_right]],
  //   specifying how many elements to discard (un-pad) from the intermediate
  //   result obtained after rearranging the batch data into spatial blocks.
  // block_size: block size for the conversion.
  // output: 4-D output tensor.
  //
  // The dimensions of the tensors are guaranteed to be correct when the
  // functor is called.
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<int32>::ConstMatrix crops,
                  int block_size, typename TTypes<T, 4>::Tensor output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHTOSPACE_OP_H_
