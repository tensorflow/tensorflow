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

#define REGISTER_CPU_SPARSE_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSum")                       \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int32>("Tidx"),            \
                          SparseSegmentReductionSumOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseSegmentSumWithNumSegments")                            \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx"),                                \
      SparseSegmentReductionSumWithNumSegmentsOp<CPUDevice, type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMean")                       \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .TypeConstraint<int32>("Tidx"),             \
                          SparseSegmentReductionMeanOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseSegmentMeanWithNumSegments")                            \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx"),                                 \
      SparseSegmentReductionMeanWithNumSegmentsOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtN")                       \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int32>("Tidx"),              \
                          SparseSegmentReductionSqrtNOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SparseSegmentSqrtNWithNumSegments")                            \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tidx"),                                  \
      SparseSegmentReductionSqrtNWithNumSegmentsOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMeanGrad")       \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentMeanGradOp<type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtNGrad")      \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentSqrtNGradOp<type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

}  // namespace tensorflow
