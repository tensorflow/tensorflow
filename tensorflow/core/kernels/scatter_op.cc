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
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

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

static void DoValidationChecking(OpKernelContext* c, const Tensor& params,
                                 const Tensor& indices, const Tensor& updates) {
  OP_REQUIRES(c, params.IsInitialized(),
              errors::FailedPrecondition("Null ref for params"));
  OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
              errors::InvalidArgument("params must be at least 1-D, got shape ",
                                      params.shape().DebugString()));
  OP_REQUIRES(
      c, ValidShapes(params, updates, indices),
      errors::InvalidArgument(
          "Must have updates.shape = indices.shape + params.shape[1:], got ",
          "updates.shape ", updates.shape().DebugString(), ", indices.shape ",
          indices.shape().DebugString(), ", params.shape ",
          params.shape().DebugString()));
}

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

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    DoValidationChecking(c, params, indices, updates);
    if (!c->status().ok()) return;

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(c, N_big <= std::numeric_limits<Index>::max(),
                errors::InvalidArgument(
                    "indices has too many elements for ",
                    DataTypeString(DataTypeToEnum<Index>::v()), " indexing: ",
                    N_big, " > ", std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(indices.NumElements());
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

    if (N > 0) {
      auto indices_flat = indices.flat<Index>();
      auto params_flat = params.flat_outer_dims<T>();
      auto updates_flat = updates.shaped<T, 2>({N, updates.NumElements() / N});

      functor::ScatterFunctor<Device, T, Index, op> functor;
      const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                  params_flat, updates_flat, indices_flat);
      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", params.dim_size(0), ")"));
    }
  }
};

namespace functor {
// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<CPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  An earlier version of the
      // code checked it and then grabbed it from memory a second time, which
      // was a security risk since it could have changed in between.
      const Index index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      Assign<op>::Run(params.template chip<0>(index),
                      updates.template chip<0>(i));
    }
    return -1;
  }
};
}  // namespace functor

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(Name(name)                                   \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

#define REGISTER_SCATTER_ADD_SUB(type, dev)                                    \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterAdd", scatter_op::UpdateOp::ADD); \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterSub", scatter_op::UpdateOp::SUB);

#define REGISTER_SCATTER_UPDATE(type, dev)            \
  REGISTER_SCATTER_KERNEL(type, dev, "ScatterUpdate", \
                          scatter_op::UpdateOp::ASSIGN);

// Registers CPU kernels.
#define REGISTER_SCATTER_ADD_SUB_CPU(type) REGISTER_SCATTER_ADD_SUB(type, CPU);

#define REGISTER_SCATTER_UPDATE_CPU(type) REGISTER_SCATTER_UPDATE(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ADD_SUB_CPU);
TF_CALL_ALL_TYPES(REGISTER_SCATTER_UPDATE_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_SCATTER_ADD_SUB_GPU(type) REGISTER_SCATTER_ADD_SUB(type, GPU);

#define REGISTER_SCATTER_UPDATE_GPU(type) REGISTER_SCATTER_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ADD_SUB_GPU);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_UPDATE_GPU);

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

#define DECLARE_GPU_SPECS_OP(T, Index, op)                   \
  template <>                                                \
  Index ScatterFunctor<GPUDevice, T, Index, op>::operator()( \
      OpKernelContext* c, const GPUDevice& d,                \
      typename TTypes<T>::Matrix params,                     \
      typename TTypes<T>::ConstMatrix updates,               \
      typename TTypes<Index>::ConstFlat indices);            \
  extern template struct ScatterFunctor<GPUDevice, T, Index, op>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)                       \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_OP

}  // namespace functor
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
