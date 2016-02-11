/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/state_ops.cc.

#include "tensorflow/core/kernels/scatter_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

template <scatter_op::UpdateOp Op>
struct Assign {};
template <>
struct Assign<scatter_op::UpdateOp::ASSIGN> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::ADD> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p += u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::SUB> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p -= u;
  }
};

}  // namespace

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
class ScatterUpdateOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here.  Should we have the framework do some sort of
  //   integer promotion automatically, or should that be something
  //   that users have to do explicitly with a conversion operator
  //   in the graph?
  explicit ScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

  // Check whether updates.shape = indices.shape + params.shape[1:]
  static bool ValidShapes(const Tensor& params, const Tensor& updates,
                          const Tensor& indices) {
    if (updates.dims() != indices.dims() + params.dims() - 1) return false;
    for (int d = 0; d < indices.dims(); d++) {
      if (updates.dim_size(d) != indices.dim_size(d)) {
        return false;
      }
    }
    for (int d = 1; d < params.dims(); d++) {
      if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
        return false;
      }
    }
    return true;
  }

  void DoCompute(OpKernelContext* c) {
    Tensor Tparams = c->mutable_input(0, use_exclusive_lock_);
    OP_REQUIRES(c, Tparams.IsInitialized(),
                errors::FailedPrecondition("Null ref for params"));
    const Tensor& Tindices = c->input(1);
    const Tensor& Tupdates = c->input(2);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(Tparams.shape()),
        errors::InvalidArgument("params must be at least 1-D, got shape ",
                                Tparams.shape().DebugString()));
    OP_REQUIRES(
        c, ValidShapes(Tparams, Tupdates, Tindices),
        errors::InvalidArgument(
            "Must have updates.shape = indices.shape + params.shape[1:], got ",
            "updates.shape ", Tupdates.shape().DebugString(),
            ", indices.shape ", Tindices.shape().DebugString(),
            ", params.shape ", Tparams.shape().DebugString()));

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

    const Index N = Tindices.NumElements();
    if (N > 0) {
      auto Tindices_flat = Tindices.flat<Index>();
      auto Tparams_flat = Tparams.flat_outer_dims<T>();
      auto Tupdates_flat =
          Tupdates.shaped<T, 2>({N, Tupdates.NumElements() / N});

      functor::ScatterFunctor<Device, T, Index, op> functor;
      functor(c, c->template eigen_device<Device>(),
              Tparams_flat, Tupdates_flat, Tindices_flat);
    }
  }
};

namespace functor {
// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<CPUDevice, T, Index, op> {
  void operator()(OpKernelContext* c, const CPUDevice& d,
                  typename TTypes<T>::Matrix params,
                  typename TTypes<T>::ConstMatrix updates,
                  typename TTypes<Index>::ConstFlat indices) {
    Index N = indices.size();
    // Validate all the indices are in range
    Index first_dim_size = params.dimension(0);
    for (Index i = 0; i < N; i++) {
      const Index index = indices(i);
      OP_REQUIRES(c, index >= 0 && index < first_dim_size,
                  errors::InvalidArgument(
                      strings::StrCat("Index ", index, " at offset ", i,
                                      " in indices is out of range")));
    }
    for (Index i = 0; i < N; i++) {
      // Copy last Ndim-1 dimensions of Tupdates[i] to
      // Tparams[Tindices[i]]
      Assign<op>::Run(params.template chip<0>(indices(i)),
                      updates.template chip<0>(i));
    }
  }
};
}  // namespace functor

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op)  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name(name)                                                        \
      .Device(DEVICE_##dev)                                             \
      .TypeConstraint<type>("T")                                        \
      .TypeConstraint<index_type>("Tindices"),                          \
      ScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)            \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op);    \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

#define REGISTER_SCATTER_ADD_SUB(type, dev)                 \
  REGISTER_SCATTER_KERNEL(                                  \
      type, dev, "ScatterAdd", scatter_op::UpdateOp::ADD);  \
  REGISTER_SCATTER_KERNEL(                                  \
      type, dev, "ScatterSub", scatter_op::UpdateOp::SUB);

#define REGISTER_SCATTER_UPDATE(type, dev)                  \
  REGISTER_SCATTER_KERNEL(                                  \
      type, dev, "ScatterUpdate", scatter_op::UpdateOp::ASSIGN);

// Registers CPU kernels.
#define REGISTER_SCATTER_ADD_SUB_CPU(type)      \
  REGISTER_SCATTER_ADD_SUB(type, CPU);

#define REGISTER_SCATTER_UPDATE_CPU(type)       \
  REGISTER_SCATTER_UPDATE(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ADD_SUB_CPU);
TF_CALL_ALL_TYPES(REGISTER_SCATTER_UPDATE_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_SCATTER_ADD_SUB_GPU(type)      \
  REGISTER_SCATTER_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_UPDATE_GPU(type)       \
  REGISTER_SCATTER_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_ADD_SUB_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SCATTER_UPDATE_GPU);

#endif  // GOOGLE_CUDA

#undef REGISTER_SCATTER_ADD
#undef REGISTER_SCATTER_ADD_SUB
#undef REGISTER_SCATTER_ADD_SUB_CPU
#undef REGISTER_SCATTER_ADD_SUB_GPU
#undef REGISTER_SCATTER_UPDATE
#undef REGISTER_SCATTER_UPDATE_CPU
#undef REGISTER_SCATTER_UPDATE_GPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPECS_OP(T, Index, op)                              \
  template <>                                                           \
  void ScatterFunctor<GPUDevice, T, Index, op>::operator()(             \
      OpKernelContext* c, const GPUDevice& d,                           \
      typename TTypes<T>::Matrix params,                                \
      typename TTypes<T>::ConstMatrix updates,                          \
      typename TTypes<Index>::ConstFlat indices);                       \
  extern template struct ScatterFunctor<GPUDevice, T, Index, op>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)                       \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);

#define DECLARE_GPU_SPECS(T)                    \
  DECLARE_GPU_SPECS_INDEX(T, int32);            \
  DECLARE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_OP

}  // namespace functor
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
