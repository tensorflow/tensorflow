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
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int64)
#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64)

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
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
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
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanGrad")                                     \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentMeanGradOp<type, index_type, segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNGrad")                                    \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSqrtNGradOp<type, index_type, segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

}  // namespace tensorflow
