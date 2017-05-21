/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/reduce_slice_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

#define GPUReduceSliceFunctorReduceop(reduceop, beginning)                     \
  template <typename T, typename Index>                                        \
  __global__ void ReduceSliceDeviceKernel##reduceop(Index sizex, Index sizey,  \
      Index sizez, Index jobsx, Index jobsy, Index jobsz, Index bound,         \
      const T begin, const Index *indices, const T *input, T *out)             \
  {                                                                            \
    Index _x = blockIdx.x * blockDim.x + threadIdx.x;                          \
    Index _y = blockIdx.y * blockDim.y + threadIdx.y;                          \
    Index _z = blockIdx.z * blockDim.z + threadIdx.z;                          \
    for(Index x = _x * jobsx; x < jobsx * (_x + 1); ++x) {                     \
      for(Index y = _y * jobsy; y < jobsy * (_y + 1); ++y) {                   \
        for(Index z = _z * jobsz; z < jobsz * (_z + 1); ++z) {                 \
          if( x < sizex && y < sizey && z < sizez ) {                          \
            Index outidx = x * sizey * sizez + y * sizez + z;                  \
            out[outidx] = begin;                                               \
            Index start = indices[y*2];                                        \
            Index end   = Min(bound, indices[y * 2 + 1]);                      \
            for(Index yin = start; yin < end; yin++) {                         \
              Index inidx = x * bound * sizez + yin * sizez + z;               \
              out[outidx] = reduceop(out[outidx], input[inidx]);               \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, typename Index>                                        \
  struct ReduceSliceFunctor##reduceop<GPUDevice, T, Index> {                   \
    virtual ~ReduceSliceFunctor##reduceop(){}                                  \
    virtual void operator()(OpKernelContext* ctx, const GPUDevice& d,          \
                            typename TTypes<Index>::ConstMatrix indices,       \
                            typename TTypes<T,3>::ConstTensor data,            \
                            typename TTypes<T,3>::Tensor output)               \
    {                                                                          \
      Index bound = data.dimension(1);                                         \
      int sizex = output.dimension(0);                                         \
      int sizey = output.dimension(1);                                         \
      int sizez = output.dimension(2);                                         \
      Cuda3DLaunchConfig config = GetCuda3DLaunchConfig(sizex, sizey, sizez, d,\
                               ReduceSliceDeviceKernel##reduceop<T, Index>, 0);\
      Index threadsx = config.thread_per_block.x * config.block_count.x;       \
      Index threadsy = config.thread_per_block.y * config.block_count.y;       \
      Index threadsz = config.thread_per_block.z * config.block_count.z;       \
      Index jobsx = (sizex + threadsx - 1) / threadsx;                         \
      Index jobsy = (sizey + threadsy - 1) / threadsy;                         \
      Index jobsz = (sizez + threadsz - 1) / threadsz;                         \
                                                                               \
      ReduceSliceDeviceKernel##reduceop<T,Index> <<<config.block_count,        \
        config.thread_per_block, 0, d.stream()>>> (sizex, sizey, sizez,        \
                                   jobsx, jobsy, jobsz, bound, beginning<T>(), \
                                   indices.data(), data.data(), output.data());\
    }                                                                          \
  };

CALL_ALL_REDUCEOPS(GPUReduceSliceFunctorReduceop)
#undef GPUReduceSliceFunctorReduceop

#define DEFINE_GPU_REDUCEOP_SPECS_INDEX(reduceop, dummy, T)                    \
  template struct ReduceSliceFunctor##reduceop<GPUDevice, T, int32>;           \
  template struct ReduceSliceFunctor##reduceop<GPUDevice, T, int64>;

#define DEFINE_GPU_SPECS(T)                                                    \
  CALL_ALL_REDUCEOPS(DEFINE_GPU_REDUCEOP_SPECS_INDEX, T)

TF_CALL_REAL_NUMBER_TYPES(DEFINE_GPU_SPECS)

#undef DEFINE_GPU_REDUCEOP_SPECS_INDEX
#undef DEFINE_GPU_SPECS

}  // namespace functor
}  // namespace tensorflow

#endif
