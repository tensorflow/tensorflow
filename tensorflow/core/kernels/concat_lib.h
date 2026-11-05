/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CONCAT_LIB_H_
#define TENSORFLOW_CORE_KERNELS_CONCAT_LIB_H_

#include <cstddef>
#include <vector>

#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// Lightweight descriptor for a flattened [dim0, dim1] input slice in CPU
// concatenation and packing kernels, avoiding per-input heap-allocated
// `Eigen::TensorMap` (`ConstMatrix`) wrappers. `data` points to `dim0 * dim1`
// contiguous elements (or may be null when `dim1 == 0`).
template <typename T>
struct ConcatSlice {
  const T* data;
  ptrdiff_t dim1;
};

// Concatenates `slices` along dimension 1 into `output_data`.
// Each slice in `slices` has outer dimension `dim0` and inner width
// `slice.dim1`, where the sum of `slice.dim1` across `slices` equals
// `row_size`. `output_data` must point to at least `dim0 * row_size` elements.
template <typename T>
void ConcatCPUSlices(DeviceBase* device,
                     absl::Span<const ConcatSlice<T>> slices, int64_t dim0,
                     int64_t row_size, T* output_data);

// Functors to concatenate tensors. These always take a rank-2 tensor (i.e a
// matrix) and concatenate it along the axis 1 ("putting them next to each
// other" as opposed to "putting them on top of one another").
//
// Any concatenation of n-dimensional tensors across any axis can be reduced to
// a concatenation of two-dimensional tensors across the axis 1 by first
// partitioning the axes of the original tensors into those less than the axis
// to be concatenated across and the rest. Then reshape the tensors into a
// two-dimensional tensor by collapsing these two sets of axes and concatenate
// the resulting matrices across the axis 1, finally reshaping the result to
// have the proper shape.
//
// So, for example, when stacking N tensors, reshape each to have shape
// {1, Numelements} and reshape the result matrix to have shape
// {1, N * NumElements} before passing it to this functor.

// Assumes all elements of inputs are nonempty.
// Assumes output is nonempty.
template <typename T>
void ConcatCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <typename T>
void ConcatGPU(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    Tensor* output, typename TTypes<T, 2>::Tensor* output_flat);

// Explicit instantiations in concat_lib_gpu.cc.
#define REGISTER(T)                                                           \
  extern template void ConcatGPU<T>(                                          \
      OpKernelContext * c,                                                    \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      Tensor* output, typename TTypes<T, 2>::Tensor* output_flat);

TF_CALL_INTEGRAL_TYPES(REGISTER);  // int32 Needed for TensorLists.
TF_CALL_GPU_ALL_TYPES(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONCAT_LIB_H_
