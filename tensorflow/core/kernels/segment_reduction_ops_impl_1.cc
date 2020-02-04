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

// See docs in ../ops/math_ops.cc.
#include "tensorflow/core/kernels/segment_reduction_ops_impl.h"

namespace tensorflow {
namespace internal {
// Static routines not in the templated class to reduce code size
void SegmentReductionValidationHelper(OpKernelContext* context,
                                      const Tensor& input,
                                      const Tensor& segment_ids) {
  OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
              errors::InvalidArgument("segment_ids should be a vector."));
  const int64 num_indices = segment_ids.NumElements();
  OP_REQUIRES(context, num_indices == input.dim_size(0),
              errors::InvalidArgument(
                  "segment_ids should be the same size as dimension 0 of"
                  " input."));
}

bool SegmentReductionDoValidation(OpKernelContext* c, const Tensor& input,
                                  const Tensor& segment_ids) {
  SegmentReductionValidationHelper(c, input, segment_ids);
  return c->status().ok();
}

// check routines not in the templated class to reduce code size
void UnsortedSegmentReductionValidation(OpKernel* op_kernel,
                                        OpKernelContext* context,
                                        const Tensor& data,
                                        const Tensor& segment_ids,
                                        const Tensor& num_segments) {
  OP_REQUIRES(
      context, TensorShapeUtils::IsScalar(num_segments.shape()),
      errors::InvalidArgument("num_segments should be a scalar, not shape ",
                              num_segments.shape().DebugString()));
  OP_REQUIRES(
      context, TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
      errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                              " does not start with segment_ids.shape = ",
                              segment_ids.shape().DebugString()));
}

bool UnsortedSegmentReductionDoValidation(OpKernel* op_kernel,
                                          OpKernelContext* context,
                                          const Tensor& data,
                                          const Tensor& segment_ids,
                                          const Tensor& num_segments) {
  UnsortedSegmentReductionValidation(op_kernel, context, data, segment_ids,
                                     num_segments);
  return context->status().ok();
}

}  // namespace internal

#define REGISTER_CPU_KERNEL_SEGMENT(name, functor, type, index_type, \
                                    default_value)                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name(name)                                                     \
          .Device(DEVICE_CPU)                                        \
          .TypeConstraint<type>("T")                                 \
          .TypeConstraint<index_type>("Tindices"),                   \
      SegmentReductionOp<CPUDevice, type, index_type, functor, default_value>)

#define REGISTER_REAL_CPU_KERNELS(type, index_type)                            \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentSum", Eigen::internal::SumReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type, 0); \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type, 1); \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentMin", Eigen::internal::MinReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentMax", Eigen::internal::MaxReducer<type>, \
                              type, index_type, 0)

#define REGISTER_COMPLEX_CPU_KERNELS(type, index_type)                         \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentSum", Eigen::internal::SumReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type, 0); \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type, 1);

#define REGISTER_REAL_CPU_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_KERNELS(type, int32)

#define REGISTER_COMPLEX_CPU_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_KERNELS(type, int32)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_KERNELS_ALL);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex128);
#undef REGISTER_CPU_KERNEL_SEGMENT
#undef REGISTER_REAL_CPU_KERNELS
#undef REGISTER_COMPLEX_CPU_KERNELS
#undef REGISTER_REAL_CPU_KERNELS_ALL
#undef REGISTER_COMPLEX_CPU_KERNELS_ALL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_SORTED_KERNELS(type, index_type)                  \
  REGISTER_KERNEL_BUILDER(Name("SegmentSum")                           \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          SegmentSumGPUOp<type, index_type>)

#define REGISTER_GPU_SORTED_KERNELS_ALL(type) \
  REGISTER_GPU_SORTED_KERNELS(type, int32)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SORTED_KERNELS_ALL);
#undef REGISTER_GPU_SORTED_KERNELS
#undef REGISTER_GPU_SORTED_KERNELS_ALL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
