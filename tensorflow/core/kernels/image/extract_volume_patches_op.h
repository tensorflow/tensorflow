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
    const int64 N = std::max(input.size(), output.size());
    if (N <= std::numeric_limits<Index32>::max()) {
      auto output_32bit = To32Bit(output);
      output_32bit.device(d) =
          To32Bit(input)
              .extract_volume_patches(patch_cols, patch_rows, patch_planes,
                                      stride_cols, stride_rows, stride_planes,
                                      padding)
              .reshape(output_32bit.dimensions());
    } else {
      output.device(d) =
          input
              .extract_volume_patches(patch_cols, patch_rows, patch_planes,
                                      stride_cols, stride_rows, stride_planes,
                                      padding)
              .reshape(output.dimensions());
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IMAGE_EXTRACT_VOLUME_PATCHES_OP_H_
