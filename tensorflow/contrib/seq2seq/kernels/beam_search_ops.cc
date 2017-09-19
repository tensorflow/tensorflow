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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/seq2seq/kernels/beam_search_ops.h"

#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class GatherTreeOp : public OpKernel {
 public:
  explicit GatherTreeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Device& device = ctx->eigen_device<Device>();
    const Tensor& step_ids = ctx->input(0);
    const Tensor& parent_ids = ctx->input(1);
    const Tensor& sequence_length = ctx->input(2);
    const TensorShape& step_ids_shape = step_ids.shape();
    OP_REQUIRES(
        ctx, step_ids_shape.dims() == 3,
        errors::InvalidArgument("step_ids must be a 3-tensor, saw shape: ",
                                step_ids_shape.DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(sequence_length.shape()),
        errors::InvalidArgument("sequence_length must be a matrix, saw shape: ",
                                sequence_length.shape().DebugString()));
    OP_REQUIRES(ctx, sequence_length.dim_size(0) == step_ids_shape.dim_size(1),
                errors::InvalidArgument(
                    "Inconsistent batch sizes: sequence_length.shape[0] (",
                    sequence_length.dim_size(0), ") != ", "step_ids.shape[1] (",
                    step_ids_shape.dim_size(1), ")"));
    OP_REQUIRES(ctx, sequence_length.dim_size(1) == step_ids_shape.dim_size(2),
                errors::InvalidArgument(
                    "Inconsistent batch sizes: sequence_length.shape[1] (",
                    sequence_length.dim_size(1), ") != ", "step_ids.shape[2] (",
                    step_ids_shape.dim_size(2), ")"));
    OP_REQUIRES(
        ctx, step_ids_shape == parent_ids.shape(),
        errors::InvalidArgument(
            "step_ids.shape must match parent_ids.shape.  but shapes are: ",
            step_ids_shape.DebugString(), " and ",
            parent_ids.shape().DebugString()));
    Tensor* beams;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, step_ids_shape, &beams));
    typename TTypes<T, 3>::ConstTensor step_ids_t = step_ids.tensor<T, 3>();
    typename TTypes<T, 3>::ConstTensor parent_ids_t = parent_ids.tensor<T, 3>();
    typename TTypes<T>::ConstMatrix seq_len_t = sequence_length.matrix<T>();
    typename TTypes<T, 3>::Tensor beams_t = beams->tensor<T, 3>();
    functor::GatherTree<Device, T>()(ctx, device, step_ids_t, parent_ids_t,
                                     seq_len_t, beams_t);
  }
};

#define REGISTER_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("GatherTree").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GatherTreeOp<CPUDevice, T>);
REGISTER_KERNEL(int32);
#undef REGISTER_KERNEL

namespace functor {

// CPU specialization
template <>
struct GatherTree<CPUDevice, int32> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  typename TTypes<int32, 3>::ConstTensor step_ids,
                  typename TTypes<int32, 3>::ConstTensor parent_ids,
                  typename TTypes<int32>::ConstMatrix sequence_length,
                  typename TTypes<int32, 3>::Tensor beams) {
    const int64 max_time = parent_ids.dimension(0);
    const int64 batch_size = parent_ids.dimension(1);
    const int64 beam_width = parent_ids.dimension(2);
    beams.setConstant(-1);

    auto DoWork = [&, ctx](int start_batch_beam, int limit_batch_beam) {
      for (int32 i = start_batch_beam; i < limit_batch_beam; ++i) {
        const int32 batch = i / beam_width;
        const int32 beam = i % beam_width;
        int32 seq_len_b = sequence_length(batch, beam);
        if (seq_len_b <= 0) {
          continue;
        }
        beams(seq_len_b - 1, batch, beam) =
            step_ids(seq_len_b - 1, batch, beam);
        int32 parent = parent_ids(seq_len_b - 1, batch, beam);
        for (int32 level = seq_len_b - 2; level >= 0; --level) {
          if (parent < 0 || parent > beam_width) {
            ctx->SetStatus(
                errors::InvalidArgument("Saw invalid parent id ", parent,
                                        " at (batch, time, beam) == (", batch,
                                        ", ", level, ", ", beam, ")"));
            return;
          }
          beams(level, batch, beam) = step_ids(level, batch, parent);
          parent = parent_ids(level, batch, parent);
        }
      }
    };
    // Guesstimate of cost; ~5 lookup/store/compare per inner beam
    // traversal time step.
    const int64 batch_beam_cost =
        Eigen::TensorOpCost::DivCost<int32>() +
        6 * Eigen::TensorOpCost::AddCost<int32>() +
        max_time * (5 * Eigen::TensorOpCost::AddCost<int32>());
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          batch_size * beam_width, batch_beam_cost, DoWork);
  }
};

}  // namespace functor

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                            \
  template <>                                          \
  void GatherTree<GPUDevice, T>::operator()(           \
      OpKernelContext* ctx, const GPUDevice& d,        \
      typename TTypes<T, 3>::ConstTensor step_ids,     \
      typename TTypes<T, 3>::ConstTensor parent_ids,   \
      typename TTypes<T>::ConstMatrix sequence_length, \
      typename TTypes<T, 3>::Tensor beams);            \
  extern template struct GatherTree<GPUDevice, T>;

DECLARE_GPU_SPEC(int32);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("GatherTree").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      GatherTreeOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(int32);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
