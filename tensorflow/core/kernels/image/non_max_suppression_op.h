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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGENON_MAX_SUPPRESSION_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMAGENON_MAX_SUPPRESSION_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
extern const int kNmsBoxesPerTread;

// Given descending sorted box list, apply non-maximal-suppression with given
// threshold and select boxes to keep.
// - d_sorted_boxes_float_ptr: a pointer to device memory float array
//   containing the box corners for N boxes sorted in descending order of
//   scores.
// - num_boxes: number of boxes.
// - iou_threshold: the intersection-over-union (iou) threshold for elimination.
// - d_selected_indices: is a device pointer to int array containing sorted
//   indices of the boxes to keep.
// - h_num_boxes_to_keep: is a host pointer for returning number of items
//   to keep.
// - flip_boxes: flag reorders the boxes use lower left and upper right
//   corners if they are given in mixed format.
Status NmsGpu(const float* d_sorted_boxes_float_ptr, const int num_boxes,
              const float iou_threshold, int* d_selected_indices,
              int* h_num_boxes_to_keep, OpKernelContext* context,
              const int max_boxes, bool flip_boxes = false);
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGENON_MAX_SUPPRESSION_OP_H_
