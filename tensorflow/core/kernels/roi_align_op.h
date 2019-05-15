/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_ROI_ALIGN_OP_H_
#define TENSORFLOW_CORE_KERNELS_ROI_ALIGN_OP_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {
template <typename Device, typename T>
struct ROIAlign {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor X,
                  typename TTypes<T, 2>::ConstTensor RoIs,
                  const int pooled_height, const int pooled_width,
                  const int samplig_ratio, const T spatial_scale,
                  typename TTypes<T, 4>::Tensor Y);
};

template <typename Device, typename T>
struct ROIAlignGrad {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor inputs,
                  typename TTypes<T, 2>::ConstTensor rois,
                  const int pooled_height, const int pooled_width,
                  const int sampling_ratio, const T spatial_scale,
                  typename TTypes<T, 4>::Tensor output);
};
}  // namespace functor
}  // namespace tensorflow

#endif
