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

#define EIGEN_USE_THREADS

#include <algorithm>
#include "tensorflow/core/kernels/reduce_slice_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using thread::ThreadPool;

namespace functor {

#if !GOOGLE_CUDA
namespace reduce_functions {

template <typename T>
inline T sum(T a,T b) { return a+b; }

template <typename T>
inline T prod(T a,T b) { return a*b; }

template <typename T>
inline T max(T a,T b) { return a>b?a:b; }

template <typename T>
inline T min(T a,T b) { return a<b?a:b; }

}
#endif // !GOOGLE_CUDA

template <typename T, typename Index, T beginning(), T reduce(T,T)>
struct ReduceSliceFunctor<CPUDevice, T, Index, beginning, reduce> {

private:
  struct XYZ {
    Index x, y, z;
    XYZ() = default;
    XYZ(Index x, Index y, Index z) : x(x), y(y), z(z) {}
  };
  inline static XYZ global_index_to_xyz(Index global,XYZ size) {
    XYZ ret;
    ret.x = global / (size.y * size.z);
    ret.y = global % (size.y * size.z) / size.z;
    ret.z = global % size.z;
    return ret;
  }

public:
  virtual ~ReduceSliceFunctor(){}
  virtual void operator()(OpKernelContext* ctx, const CPUDevice& d,
                          typename TTypes<Index>::ConstMatrix indices,
                          typename TTypes<T,3>::ConstTensor data,
                          typename TTypes<T,3>::Tensor output)
  {
    Index bound = data.dimension(1);
    Index dim1 = output.dimension(0);
    Index dim2 = output.dimension(1);
    Index dim3 = output.dimension(2);
    T zero = beginning();
    ThreadPool* thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    // shard the work
    auto work = [&](Index start, Index end) {
      for(Index global = start; global < end; ++global) {
        XYZ xyz = global_index_to_xyz(global, XYZ(dim1, dim2, dim3));
        Index x = xyz.x;
        Index y = xyz.y;
        Index z = xyz.z;
        output(x, y, z) = zero;
        Index slice_head = indices(y, 0);
        Index slice_end = std::min(indices(y,1), bound);
        for(Index i = slice_head; i < slice_end; ++i) {
          output(x, y, z) = reduce(output(x, y, z), data(x, i, z));
        }
      }
    };
    // Here assumes the number of average CPU cycles for each slice equals the
    // average length of each slice
    thread_pool->ParallelFor(dim1*dim2*dim3,std::max(bound/dim2, (Index)1), work);
  }
};

#define DEFINE_CPU_SUMPROD_SPECS_INDEX(T, Index)                               \
  template struct ReduceSliceFunctor<CPUDevice, T, Index,                      \
           reduce_functions::zero<T>, reduce_functions::sum<T>>;               \
  template struct ReduceSliceFunctor<CPUDevice, T, Index,                      \
           reduce_functions::one<T>, reduce_functions::prod<T>>;

#define DEFINE_CPU_MINMAX_SPECS_INDEX(T, Index)                                \
  template struct ReduceSliceFunctor<CPUDevice, T, Index,                      \
           reduce_functions::negative_infinity<T>, reduce_functions::max<T>>;  \
  template struct ReduceSliceFunctor<CPUDevice, T, Index,                      \
           reduce_functions::infinity<T>, reduce_functions::min<T>>;

#define DEFINE_CPU_SUMPROD_SPECS(T)          \
  DEFINE_CPU_SUMPROD_SPECS_INDEX(T, int32);  \
  DEFINE_CPU_SUMPROD_SPECS_INDEX(T, int64);

#define DEFINE_CPU_MINMAX_SPECS(T)          \
  DEFINE_CPU_MINMAX_SPECS_INDEX(T, int32);  \
  DEFINE_CPU_MINMAX_SPECS_INDEX(T, int64);

TF_CALL_NUMBER_TYPES(DEFINE_CPU_SUMPROD_SPECS)
TF_CALL_REAL_NUMBER_TYPES(DEFINE_CPU_MINMAX_SPECS)

#undef DEFINE_CPU_SUMPROD_SPECS_INDEX
#undef DEFINE_CPU_MINMAX_SPECS_INDEX
#undef DEFINE_CPU_SUMPROD_SPECS
#undef DEFINE_CPU_MINMAX_SPECS

} // namespace functor

template <typename Device, typename T, typename Index, T beginning(), T reduce(T,T)>
class ReduceSliceKernel : public OpKernel {
public:
  explicit ReduceSliceKernel(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor &data = context->input(0);
    const Tensor &indices = context->input(1);
    const Tensor &_axis = context->input(2);
    int64 axis = _axis.scalar<int64>()(0);
    TensorShape output_shape = data.shape();
    output_shape.set_dim(axis,indices.shape().dim_size(0));
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto functor = functor::ReduceSliceFunctor<Device, T, Index, beginning, reduce>();
    functor(context, context->eigen_device<Device>(), indices.matrix<Index>(),
        data.flat_inner_outer_dims<T,3>(axis-1), output->flat_inner_outer_dims<T,3>(axis-1));
  }
};

#if GOOGLE_CUDA

#define REGISTER_GPU_REDUCE_SLICE_KERNELS(type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceSum")                               \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<GPUDevice, type, index_type,       \
                          functor::reduce_functions::zero<type>,               \
                          functor::reduce_functions::sum<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceProd")                              \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<GPUDevice, type, index_type,       \
                          functor::reduce_functions::one<type>,                \
                          functor::reduce_functions::prod<type>>);             \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceMax")                               \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<GPUDevice, type, index_type,       \
                          functor::reduce_functions::negative_infinity<type>,  \
                          functor::reduce_functions::max<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceMin")                               \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<GPUDevice, type, index_type,       \
                          functor::reduce_functions::infinity<type>,           \
                          functor::reduce_functions::min<type>>);

#define REGISTER_GPU_REDUCE_SLICE_KERNELS_ALL(type) \
  REGISTER_GPU_REDUCE_SLICE_KERNELS(type, int32);   \
  REGISTER_GPU_REDUCE_SLICE_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_REDUCE_SLICE_KERNELS_ALL);

#undef REGISTER_GPU_REDUCE_SLICE_KERNELS
#undef REGISTER_GPU_REDUCE_SLICE_KERNELS_ALL

#endif  // GOOGLE_CUDA

#define REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS(type, index_type)            \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceSum")                               \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<CPUDevice, type, index_type,       \
                          functor::reduce_functions::zero<type>,               \
                          functor::reduce_functions::sum<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceProd")                              \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<CPUDevice, type, index_type,       \
                          functor::reduce_functions::one<type>,                \
                          functor::reduce_functions::prod<type>>);

#define REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS(type, index_type)             \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceMax")                               \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<CPUDevice, type, index_type,       \
                          functor::reduce_functions::negative_infinity<type>,  \
                          functor::reduce_functions::max<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("ReduceSliceMin")                               \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          ReduceSliceKernel<CPUDevice, type, index_type,       \
                          functor::reduce_functions::infinity<type>,           \
                          functor::reduce_functions::min<type>>);

#define REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS_ALL(type)           \
  REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS(type, int32);             \
  REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS(type, int64);

#define REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS_ALL(type)           \
  REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS(type, int32);             \
  REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS_ALL)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS_ALL)

#undef REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS
#undef REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS
#undef REGISTER_CPU_SUMPROD_REDUCE_SLICE_KERNELS_ALL
#undef REGISTER_CPU_MINMAX_REDUCE_SLICE_KERNELS_ALL

}  // namespace tensorflow
