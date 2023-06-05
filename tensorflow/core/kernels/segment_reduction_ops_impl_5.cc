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

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int64_t)
#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64_t)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type)       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSum")                                                \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumOp<CPUDevice, type, index_type,                \
                                  segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSumWithNumSegments")                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumWithNumSegmentsOp<CPUDevice, type, index_type, \
                                                 segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type)        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMean")                                                \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanOp<CPUDevice, type, index_type,                \
                                   segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMeanWithNumSegments")                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanWithNumSegmentsOp<CPUDevice, type, index_type, \
                                                  segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtN")                                        \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNOp<CPUDevice, type, index_type,        \
                                    segment_ids_type>);                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNWithNumSegments")                         \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNWithNumSegmentsOp<                     \
          CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#if GOOGLE_CUDA

#define REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_GPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_GPU_SPARSE_KERNELS(type, index_type, int64_t)
#define REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64_t)

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type)       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSum")                                                \
          .Device(DEVICE_GPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumOp<GPUDevice, type, index_type,                \
                                  segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSumWithNumSegments")                                 \
          .Device(DEVICE_GPU)                                                 \
          .HostMemory("num_segments")                                         \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumWithNumSegmentsOp<GPUDevice, type, index_type, \
                                                 segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_GPU_SPARSE_KERNELS

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type)        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMean")                                                \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanOp<GPUDevice, type, index_type,                \
                                   segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMeanWithNumSegments")                                 \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("num_segments")                                          \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanWithNumSegmentsOp<GPUDevice, type, index_type, \
                                                  segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_GPU_SPARSE_KERNELS

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtN")                                        \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNOp<GPUDevice, type, index_type,        \
                                    segment_ids_type>);                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNWithNumSegments")                         \
          .Device(DEVICE_GPU)                                           \
          .HostMemory("num_segments")                                   \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNWithNumSegmentsOp<                     \
          GPUDevice, type, index_type, segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_GPU_SPARSE_KERNELS

#endif  // GOOGLE_CUDA

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSumGrad")                                      \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSumGradOp<CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanGrad")                                     \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentMeanGradOp<CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNGrad")                                    \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSqrtNGradOp<CPUDevice, type, index_type,             \
                               segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

#if GOOGLE_CUDA

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSumGrad")                                      \
          .Device(DEVICE_GPU)                                           \
          .HostMemory("output_dim0")                                    \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSumGradOp<GPUDevice, type, index_type, segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_GPU_SPARSE_KERNELS

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanGrad")                                     \
          .Device(DEVICE_GPU)                                           \
          .HostMemory("output_dim0")                                    \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentMeanGradOp<GPUDevice, type, index_type, segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_GPU_SPARSE_KERNELS

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNGrad")                                    \
          .Device(DEVICE_GPU)                                           \
          .HostMemory("output_dim0")                                    \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSqrtNGradOp<GPUDevice, type, index_type,             \
                               segment_ids_type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_GPU_SPARSE_KERNELS

#undef REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_GPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
