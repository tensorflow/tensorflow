/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_IMAGE_EXTRACT_VOLUME_PATCHES_OP_H_
#define TENSORFLOW_KERNELS_IMAGE_EXTRACT_VOLUME_PATCHES_OP_H_

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct ExtractVolumePatchesForward {
  void operator()(const Device& d, typename TTypes<T, 5>::ConstTensor input,
                  int patch_planes, int patch_rows, int patch_cols,
                  int stride_planes, int stride_rows, int stride_cols,
                  /* int rate_planes, int rate_rows, int rate_cols, */
                  const Eigen::PaddingType& padding,
                  typename TTypes<T, 5>::Tensor output) {
    MaybeWith32BitIndexing<Device>(
        [&](auto input32, auto output32) {
          output32.device(d) =
              input32
                  .extract_volume_patches(patch_cols, patch_rows, patch_planes,
                                          stride_cols, stride_rows,
                                          stride_planes, padding)
                  .reshape(output32.dimensions());
        },
        input, output);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IMAGE_EXTRACT_VOLUME_PATCHES_OP_H_
