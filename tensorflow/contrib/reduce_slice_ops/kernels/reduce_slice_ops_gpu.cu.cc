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

#include "tensorflow/contrib/reduce_slice_ops/kernels/reduce_slice_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

#define Sum(a, b) ((a) + (b))
#define Prod(a, b) ((a) * (b))
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) < (b) ? (a) : (b))

#define GPUReduceSliceFunctorReduceop(reduceop, beginning)                     \
  template <typename T, typename Index>                                        \
  __global__ void ReduceSliceDeviceKernel##reduceop(                           \
      Cuda3DLaunchConfig config, Index indices_width, Index bound,             \
      const T begin, const Index *indices, const T *input, T *out) {           \
    CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {               \
      CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {             \
        CUDA_AXIS_KERNEL_LOOP(z, config.virtual_thread_count.z, Z) {           \
          Index outidx = x * config.virtual_thread_count.y *                   \
                             config.virtual_thread_count.z +                   \
                         y * config.virtual_thread_count.z + z;                \
          out[outidx] = begin;                                                 \
          Index start = indices[y * indices_width];                            \
          Index end = Min(bound, indices[y * indices_width + 1]);              \
          for (Index yin = start; yin < end; yin++) {                          \
            Index inidx = x * bound * config.virtual_thread_count.z +          \
                          yin * config.virtual_thread_count.z + z;             \
            out[outidx] = reduceop(out[outidx], input[inidx]);                 \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, typename Index>                                        \
  struct ReduceSliceFunctor##reduceop<GPUDevice, T, Index> {                   \
    virtual ~ReduceSliceFunctor##reduceop() {}                                 \
    virtual void operator()(OpKernelContext *ctx, const GPUDevice &d,          \
                            Index indices_width,                               \
                            typename TTypes<Index, 1>::ConstTensor indices,    \
                            typename TTypes<T, 3>::ConstTensor data,           \
                            typename TTypes<T, 3>::Tensor output) {            \
      Index bound = data.dimension(1);                                         \
      int sizex = output.dimension(0);                                         \
      int sizey = output.dimension(1);                                         \
      int sizez = output.dimension(2);                                         \
      if (sizex * sizey * sizez == 0) {                                        \
        return;                                                                \
      }                                                                        \
      Cuda3DLaunchConfig config = GetCuda3DLaunchConfig(                       \
          sizex, sizey, sizez, d, ReduceSliceDeviceKernel##reduceop<T, Index>, \
          0, 0);                                                               \
                                                                               \
      ReduceSliceDeviceKernel##reduceop<T, Index>                              \
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(    \
              config, indices_width, bound, beginning<T>(), indices.data(),    \
              data.data(), output.data());                                     \
    }                                                                          \
  };

CALL_ALL_REDUCEOPS(GPUReduceSliceFunctorReduceop)
#undef GPUReduceSliceFunctorReduceop

#define DEFINE_GPU_REDUCEOP_SPECS_INDEX(reduceop, dummy, T)          \
  template struct ReduceSliceFunctor##reduceop<GPUDevice, T, int32>; \
  template struct ReduceSliceFunctor##reduceop<GPUDevice, T, int64>;

#define DEFINE_GPU_SPECS(T) \
  CALL_ALL_REDUCEOPS(DEFINE_GPU_REDUCEOP_SPECS_INDEX, T)

TF_CALL_REAL_NUMBER_TYPES(DEFINE_GPU_SPECS)

#undef DEFINE_GPU_REDUCEOP_SPECS_INDEX
#undef DEFINE_GPU_SPECS

#undef Sum
#undef Prod
#undef Min
#undef Max

}  // namespace functor
}  // namespace tensorflow

#endif
