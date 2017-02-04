/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/dense_update_ops.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_KERNEL(Var);

template <typename Device, typename T>
class ReadVariableOp : public OpKernel {
 public:
  ReadVariableOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) {
    Var* variable = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &variable));
    core::ScopedUnref s(variable);
    // TODO(apassos): It's possible to do copy-on-write here instead of always
    // copying by coordinating with the writing code. Do this. This will also
    // obviate the need to hold a lock here.
    mutex_lock ml(*variable->mu());
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, variable->tensor()->shape(), &out));
    functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
    const Tensor& t = *variable->tensor();
    copy_functor(ctx->eigen_device<Device>(), out->flat<T>(), t.flat<T>());
  }
};

class DestroyResourceOp : public OpKernel {
 public:
  explicit DestroyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
  }

  void Compute(OpKernelContext* ctx) override {
    const ResourceHandle& p = HandleFromInput(ctx, 0);
    Status status = DeleteResource(ctx, p);
    if (ignore_lookup_error_ && errors::IsNotFound(status)) {
      return;
    }
    OP_REQUIRES_OK(ctx, status);
  }

 private:
  bool ignore_lookup_error_;
};

REGISTER_KERNEL_BUILDER(Name("DestroyResourceOp").Device(DEVICE_CPU),
                        DestroyResourceOp);

// TODO(apassos) register for the GPU as well.
#define REGISTER_KERNELS(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ReadVariableOp").Device(DEVICE_CPU).TypeConstraint<type>("dtype"), \
      ReadVariableOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                                             \
  namespace functor {                                                          \
  template <>                                                                  \
  void DenseUpdate<GPUDevice, type, ASSIGN>::operator()(                       \
      const GPUDevice& d, typename TTypes<type>::Flat lhs,                     \
      typename TTypes<type>::ConstFlat rhs);                                   \
  extern template struct DenseUpdate<GPUDevice, type, ASSIGN>;                 \
  }                                                                            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ReadVariableOp").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      ReadVariableOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class AssignVariableOp : public OpKernel {
 public:
  AssignVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    Var* variable = nullptr;
    OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<Var>(
            context, HandleFromInput(context, 0), &variable,
            [this, context](Var** ptr) {
              *ptr = new Var(dtype_);
              PersistentTensor unused;
              Tensor* tmp;
              AllocatorAttributes attr;
              attr.set_gpu_compatible(true);
              attr.set_nic_compatible(true);
              TF_RETURN_IF_ERROR(context->allocate_persistent(
                  dtype_, context->input(1).shape(), &unused, &tmp, attr));
              *(*ptr)->tensor() = *tmp;
              return Status::OK();
            }));
    core::ScopedUnref s(variable);

    // TODO(apassos): holding a lock and copying is unnecessary if we are the
    // last user of the value tensor. This should essentially always be the
    // case, yet the refcount is usually 2 instead of 1. Figure out what needs
    // to change in the code to make this not be the case, so we can safely take
    // ownership.
    mutex_lock ml(*variable->mu());
    const Tensor& value = context->input(1);
    functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
    copy_functor(context->eigen_device<Device>(), variable->tensor()->flat<T>(),
                 value.flat<T>());
  }

 private:
  DataType dtype_;
};

// TODO(apassos) register for the GPU as well.
#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")            \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          AssignVariableOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")            \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("dtype"), \
                          AssignVariableOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

template <typename Device, typename T, DenseUpdateType Op>
class AssignUpdateVariableOp : public OpKernel {
 public:
  explicit AssignUpdateVariableOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    Var* variable = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &variable));
    core::ScopedUnref s(variable);

    // TODO(apassos): holding a lock and copying is unnecessary if we are the
    // last user of the value tensor. This should essentially always be the
    // case, yet the refcount is usually 2 instead of 1. Figure out what needs
    // to change in the code to make this not be the case, so we can safely take
    // ownership.
    mutex_lock ml(*variable->mu());
    const Tensor& value = context->input(1);
    functor::DenseUpdate<Device, T, Op> update_functor;
    update_functor(context->eigen_device<Device>(),
                   variable->tensor()->flat<T>(), value.flat<T>());
  }
};

#define REGISTER_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AssignAddVariableOp")                                  \
          .Device(DEVICE_CPU)                                      \
          .TypeConstraint<type>("dtype"),                          \
      AssignUpdateVariableOp<Eigen::ThreadPoolDevice, type, ADD>); \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AssignSubVariableOp")                                  \
          .Device(DEVICE_CPU)                                      \
          .TypeConstraint<type>("dtype"),                          \
      AssignUpdateVariableOp<Eigen::ThreadPoolDevice, type, SUB>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                                       \
  namespace functor {                                                    \
  template <>                                                            \
  void DenseUpdate<GPUDevice, type, ADD>::operator()(                    \
      const GPUDevice& d, typename TTypes<type>::Flat lhs,               \
      typename TTypes<type>::ConstFlat rhs);                             \
  extern template struct DenseUpdate<GPUDevice, type, ADD>;              \
  }                                                                      \
  namespace functor {                                                    \
  template <>                                                            \
  void DenseUpdate<GPUDevice, type, SUB>::operator()(                    \
      const GPUDevice& d, typename TTypes<type>::Flat lhs,               \
      typename TTypes<type>::ConstFlat rhs);                             \
  extern template struct DenseUpdate<GPUDevice, type, SUB>;              \
  }                                                                      \
  REGISTER_KERNEL_BUILDER(Name("AssignAddVariableOp")                    \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("dtype"),            \
                          AssignUpdateVariableOp<GPUDevice, type, ADD>); \
  REGISTER_KERNEL_BUILDER(Name("AssignSubVariableOp")                    \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("dtype"),            \
                          AssignUpdateVariableOp<GPUDevice, type, SUB>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp").Device(DEVICE_CPU),
                        IsResourceInitialized<Var>);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp").Device(DEVICE_GPU),
                        IsResourceInitialized<Var>);

#endif  // GOOGLE_CUDA

template <typename Device, typename T, typename Index>
class ResourceGatherOp : public OpKernel {
 public:
  explicit ResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Var* v = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    const Tensor& params = *v->tensor();
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // Check that we have enough index space
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is indices.shape + params.shape[1:].
    TensorShape result_shape = indices.shape();
    for (int i = 1; i < params.dims(); i++) {
      result_shape.AddDim(params.dim_size(i));
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N > 0) {
      auto params_flat = params.flat_outer_dims<T>();
      auto indices_flat = indices.flat<Index>();
      auto out_flat = out->shaped<T, 2>({N, out->NumElements() / N});

      functor::GatherFunctor<Device, T, Index> functor;
      int64 bad_i = functor(c->eigen_device<Device>(), params_flat,
                            indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", params.dim_size(0), ")"));
    }
  }
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                       \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          ResourceGatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
class ResourceScatterUpdateOp : public OpKernel {
 public:
  explicit ResourceScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Var* v = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(c, N_big <= std::numeric_limits<Index>::max(),
                errors::InvalidArgument(
                    "indices has too many elements for ",
                    DataTypeString(DataTypeToEnum<Index>::v()), " indexing: ",
                    N_big, " > ", std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(indices.NumElements());
    OP_REQUIRES(
        c, params->dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params->dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    if (N > 0) {
      auto indices_flat = indices.flat<Index>();
      auto params_flat = params->flat_outer_dims<T>();
      auto updates_flat = updates.shaped<T, 2>({N, updates.NumElements() / N});

      functor::ScatterFunctor<Device, T, Index, op> functor;
      const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                  params_flat, updates_flat, indices_flat);
      OP_REQUIRES(c, bad_i < 0,
                  errors::InvalidArgument(
                      "indices", SliceDebugString(indices.shape(), bad_i),
                      " = ", indices_flat(bad_i), " is not in [0, ",
                      params->dim_size(0), ")"));
    }
  }
};

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name)                                                       \
          .Device(DEVICE_##dev)                                        \
          .TypeConstraint<type>("dtype")                               \
          .TypeConstraint<index_type>("Tindices"),                     \
      ResourceScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

// TODO(apassos) add the other types here.
#define REGISTER_SCATTER_ARITHEMTIC(type, dev)             \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterAdd", \
                          scatter_op::UpdateOp::ADD);

// Registers CPU kernels.
#define REGISTER_SCATTER_ARITHEMTIC_CPU(type) \
  REGISTER_SCATTER_ARITHEMTIC(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ARITHEMTIC_CPU);

#undef REGISTER_SCATTER_ARITHEMTIC
#undef REGISTER_SCATTER_ARITHEMTIC_CPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX

}  // namespace tensorflow
