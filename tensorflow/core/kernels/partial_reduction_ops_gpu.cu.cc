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

#include "tensorflow/core/kernels/partial_reduction_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename T>
__device__ T sum_gpu(T a,T b) { return a+b; }

template <typename T>
__device__ T zero_gpu() { return T(0); }

// Kernel to do the reducton:
// x is row index of output
// y is column index
template <typename T, typename Index, T beginning(), T reduce(T,T)>
__global__ void PartialReduceKernel(Index num_rows, Index num_cols,
    const Index *indices, const T *input, T *out)
{
  Index x = blockIdx.x * blockDim.x + threadIdx.x;
  Index y = blockIdx.y * blockDim.y + threadIdx.y;
  Index outidx = x*num_cols + y;
  if( x<num_rows && y<num_cols ) {
    out[outidx] = beginning();
    Index start = indices[x*2];
    Index end   = indices[x*2+1];
    for(Index j=start;j<end;j++) {
      Index inidx = j*num_cols + y;
      out[outidx] = reduce(out[outidx],input[inidx]);
    }
  }
}

namespace functor {

template <typename T, typename Index, T beginning(), T reduce(T,T)>
struct PartialReductionFunctor<GPUDevice, T, Index, beginning, reduce>{
  virtual ~PartialReductionFunctor(){}
  virtual void operator()(
      OpKernelContext* ctx, const GPUDevice& d,typename TTypes<Index>::ConstMatrix indices,
      typename TTypes<T>::ConstMatrix data,typename TTypes<T>::Matrix output)
  {
    Index rows = output.dimension(0);
    Index cols = output.dimension(1);
    Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(rows,cols,d);
    PartialReduceKernel<T,Index,beginning,reduce><<<config.block_count,
      config.thread_per_block, 0, d.stream()>>>(rows, cols, indices.data(), data.data(), output.data());
  }
};

#define DEFINE_GPU_SPECS_INDEX(T, Index) \
  template struct PartialReductionFunctor<GPUDevice, T, Index, zero_gpu<T>, sum_gpu<T>>

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_REAL_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#endif
