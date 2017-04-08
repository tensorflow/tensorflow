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

#include "tensorflow/core/kernels/partial_reduction_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using thread::ThreadPool;

namespace functor {

template <typename T, typename Index, T beginning(), T reduce(T,T)>
struct PartialReductionFunctor<CPUDevice, T, Index, beginning, reduce>{
  virtual ~PartialReductionFunctor(){}
  virtual void operator()(
      OpKernelContext* ctx, const CPUDevice& d,typename TTypes<Index>::ConstMatrix indices,
      typename TTypes<T>::ConstMatrix data,typename TTypes<T>::Matrix output)
  {
    Index bound = data.dimension(0);
    Index rows = output.dimension(0);
    Index cols = output.dimension(1);
    T zero = beginning();
    ThreadPool* thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    // shard the work by columns
    auto work = [&](Index start, Index end) {
      // each row
      for(Index r=0;r<rows;r++) {
        // each column
        for(Index c=start;c<end;c++) {
          output(r,c) = zero;
          Index cl = indices(r,0);
          Index cr = std::min(indices(r,1),bound);
          for(Index rcursor=cl;rcursor<cr;rcursor++)
            output(r,c) = reduce(output(r,c),data(rcursor,c));
        }
      }
    };
    thread_pool->ParallelFor(cols,rows,work);
  }
};

#define DEFINE_CPU_SPECS_INDEX(T, Index)                                       \
  template struct PartialReductionFunctor<CPUDevice, T, Index,                 \
           reduce_functions::zero<T>, reduce_functions::sum<T>>;               \
  template struct PartialReductionFunctor<CPUDevice, T, Index,                 \
           reduce_functions::one<T>, reduce_functions::prod<T>>;               \
  template struct PartialReductionFunctor<CPUDevice, T, Index,                 \
           reduce_functions::negative_infinity<T>, reduce_functions::max<T>>;  \
  template struct PartialReductionFunctor<CPUDevice, T, Index,                 \
           reduce_functions::infinity<T>, reduce_functions::min<T>>;
#define DEFINE_CPU_SPECS(T)          \
  DEFINE_CPU_SPECS_INDEX(T, int32);  \
  DEFINE_CPU_SPECS_INDEX(T, int64);

TF_CALL_NUMBER_TYPES(DEFINE_CPU_SPECS)

#undef DEFINE_CPU_SPECS
#undef DEFINE_CPU_SPECS_INDEX

} // namespace functor


template <typename Device, typename T, typename Index, T beginning(), T reduce(T,T)>
class PartialReduce : public OpKernel {
public:
  explicit PartialReduce(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor &indices = context->input(1);
    const Tensor &data = context->input(0);
    TensorShape output_shape = data.shape();
    output_shape.set_dim(0,indices.shape().dim_size(0));
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto functor = functor::PartialReductionFunctor<Device, T, Index, beginning, reduce>();
    functor(context, context->eigen_device<Device>(), indices.matrix<Index>(), data.matrix<T>(), output->matrix<T>());
  }
};

#if GOOGLE_CUDA

#define REGISTER_GPU_PARTIAL_REDUCE_KERNELS(type, index_type)                  \
  REGISTER_KERNEL_BUILDER(Name("PartialSum")                                   \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<GPUDevice, type, index_type,           \
                          functor::reduce_functions::zero<type>,               \
                          functor::reduce_functions::sum<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("PartialProd")                                  \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<GPUDevice, type, index_type,           \
                          functor::reduce_functions::one<type>,                \
                          functor::reduce_functions::prod<type>>);             \
  REGISTER_KERNEL_BUILDER(Name("PartialMax")                                   \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<GPUDevice, type, index_type,           \
                          functor::reduce_functions::negative_infinity<type>,  \
                          functor::reduce_functions::max<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("PartialMin")                                   \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<GPUDevice, type, index_type,           \
                          functor::reduce_functions::infinity<type>,           \
                          functor::reduce_functions::min<type>>);

#define REGISTER_GPU_PARTIAL_REDUCE_KERNELS_ALL(type) \
  REGISTER_GPU_PARTIAL_REDUCE_KERNELS(type, int32);   \
  REGISTER_GPU_PARTIAL_REDUCE_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_PARTIAL_REDUCE_KERNELS_ALL);

#undef REGISTER_GPU_PARTIAL_REDUCE_KERNELS
#undef REGISTER_GPU_PARTIAL_REDUCE_KERNELS_ALL

#endif  // GOOGLE_CUDA

#define REGISTER_CPU_PARTIAL_REDUCE_KERNELS(type, index_type)                  \
  REGISTER_KERNEL_BUILDER(Name("PartialSum")                                   \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<CPUDevice, type, index_type,           \
                          functor::reduce_functions::zero<type>,               \
                          functor::reduce_functions::sum<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("PartialProd")                                  \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<CPUDevice, type, index_type,           \
                          functor::reduce_functions::one<type>,                \
                          functor::reduce_functions::prod<type>>);             \
  REGISTER_KERNEL_BUILDER(Name("PartialMax")                                   \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<CPUDevice, type, index_type,           \
                          functor::reduce_functions::negative_infinity<type>,  \
                          functor::reduce_functions::max<type>>);              \
  REGISTER_KERNEL_BUILDER(Name("PartialMin")                                   \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<index_type>("Tindices"),         \
                          PartialReduce<CPUDevice, type, index_type,           \
                          functor::reduce_functions::infinity<type>,           \
                          functor::reduce_functions::min<type>>);

#define REGISTER_CPU_PARTIAL_REDUCE_KERNELS_ALL(type) \
  REGISTER_CPU_PARTIAL_REDUCE_KERNELS(type, int32);   \
  REGISTER_CPU_PARTIAL_REDUCE_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_PARTIAL_REDUCE_KERNELS_ALL);

#undef REGISTER_CPU_PARTIAL_REDUCE_KERNELS
#undef REGISTER_CPU_PARTIAL_REDUCE_KERNELS_ALL


}  // namespace tensorflow
