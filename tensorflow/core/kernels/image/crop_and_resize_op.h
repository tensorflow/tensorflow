/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_CROP_AND_RESIZE_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_CROP_AND_RESIZE_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct CropAndResize {
  // We assume that the tensor sizes are correct.
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  const std::string& method_name, float extrapolation_value,
                  typename TTypes<float, 4>::Tensor crops);
};

template <typename Device, typename T>
struct CropAndResizeBackpropImage {
  // We assume that the tensor sizes are correct.
  bool operator()(const OpKernelContext* context,
                  typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 4>::Tensor grads_image,
                  const std::string& method_name);
};

template <typename Device, typename T>
struct CropAndResizeBackpropBoxes {
  // We assume that the tensor sizes are correct.
  bool operator()(const Device& d, typename TTypes<float, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<float, 2>::Tensor grads_boxes);
};

template <typename Device>
struct CheckValidBoxIndexHelper {
  // Checks if all values in box_index are in [0, batch).
  void operator()(const Device& d,
                  typename TTypes<int32, 1>::ConstTensor box_index, int batch,
                  typename TTypes<bool, 0>::Tensor isvalid) {
    isvalid.device(d) = ((box_index >= 0) && (box_index < batch)).all();
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_CROP_AND_RESIZE_OP_H_
