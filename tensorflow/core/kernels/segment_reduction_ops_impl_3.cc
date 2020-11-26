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

#define REGISTER_CPU_KERNEL_UNSORTEDSEGMENT(                           \
    name, type, index_type, initial_value_functor, reduction_functor)  \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name)                                                       \
          .Device(DEVICE_CPU)                                          \
          .TypeConstraint<type>("T")                                   \
          .TypeConstraint<index_type>("Tindices"),                     \
      UnsortedSegmentReductionOp<                                      \
          type, index_type,                                            \
          functor::UnsortedSegmentFunctor<CPUDevice, type, index_type, \
                                          initial_value_functor,       \
                                          reduction_functor> >)

#define REGISTER_REAL_CPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type,  \
                                      functor::Zero<type>,                     \
                                      functor::SumOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMax", type, index_type,  \
                                      functor::Lowest<type>,                   \
                                      functor::MaxOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMin", type, index_type,  \
                                      functor::Highest<type>,                  \
                                      functor::MinOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      functor::One<type>,                      \
                                      functor::ProdOp<type>);

#define REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, index_type)                \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type,  \
                                      functor::Zero<type>,                     \
                                      functor::SumOp<type>);                   \
  REGISTER_CPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      functor::One<type>,                      \
                                      functor::ProdOp<type>)

#define REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_UNSORTED_KERNELS(type, int32)

#define REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, int32)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL);
REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(complex128);

#undef REGISTER_REAL_CPU_UNSORTED_KERNELS
#undef REGISTER_CPU_KERNEL_UNSORTEDSEGMENT
#undef REGISTER_COMPLEX_CPU_UNSORTED_KERNELS
#undef REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL
#undef REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                 \
    name, type, index_type, initial_value_functor, reduction_kernel_functor) \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_GPU)                                                \
          .HostMemory("num_segments")                                        \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      UnsortedSegmentReductionOp<                                            \
          type, index_type,                                                  \
          functor::UnsortedSegmentFunctor<GPUDevice, type, index_type,       \
                                          initial_value_functor,             \
                                          reduction_kernel_functor> >)

// sum is the only op that supports all input types currently
#define REGISTER_REAL_GPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMax", type, index_type,  \
                                      functor::Lowest<type>,                   \
                                      functor::AtomicMaxOpGpu<type>);          \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMin", type, index_type,  \
                                      functor::Highest<type>,                  \
                                      functor::AtomicMinOpGpu<type>);          \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      functor::One<type>,                      \
                                      functor::AtomicProdOpGpu<type>);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type, \
                                      functor::Zero<type>,                    \
                                      functor::AtomicSumOpGpu<type>);

#define REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, int32)

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, int32)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
// TODO(rocm): support atomicAdd for complex numbers on ROCm
#if GOOGLE_CUDA
TF_CALL_COMPLEX_TYPES(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
#endif

#undef REGISTER_GPU_KERNEL_UNSORTEDSEGMENT
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
