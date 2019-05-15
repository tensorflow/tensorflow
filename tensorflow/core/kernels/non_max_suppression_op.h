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

#ifndef TENSORFLOW_CORE_KERNELS_NON_MAX_SUPPRESSION_OP_H_
#define TENSORFLOW_CORE_KERNELS_NON_MAX_SUPPRESSION_OP_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct NonMaxSuppression {
  void operator()(const Device& d, typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<float, 1>::ConstTensor scores,
                  float iou_threshold, float score_threshold,
                  int max_output_size,
                  typename TTypes<int, 1>::Tensor selected_indices);
};
}  // namespace functor
#if GOOGLE_CUDA
extern const int NMS_BOXES_PER_THREAD;
tensorflow::Status nms_gpu(const float* d_desc_sorted_boxes_float_ptr,
                           const int N, const float thresh,
                           int* d_keep_sorted_list, int* h_nkeep,
                           int* dev_delete_mask, int* host_delete_mask,
                           OpKernelContext* context, bool flip_boxes = false);
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_NON_MAX_SUPPRESSION_OP_H_
