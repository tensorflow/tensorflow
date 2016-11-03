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
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_KERNEL(Var);

template <typename Device, typename T>
class CreateVariableOp : public OpKernel {
 public:
  CreateVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES(c, DataTypeToEnum<T>::value == dtype_,
                errors::InvalidArgument(
                    "Dtypes don't match; expected ", DataTypeString(dtype_),
                    " got ", DataTypeString(DataTypeToEnum<T>::value)));
  }

  void Compute(OpKernelContext* context) override {
    Var* var = new Var(dtype_);
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    PersistentTensor copy;
    Tensor value = context->input(1);

    // TODO(apassos): allocating and copying is unnecessary if we are the last
    // user of the value tensor. This should essentially always be the case, yet
    // the refcount is usually 2 instead of 1. Figure out what needs to change
    // in the code to make this not be the case, so we can safely take
    // ownership.
    Tensor* tmp_copy = nullptr;
    OP_REQUIRES_OK(context, context->allocate_persistent(
                                dtype_, value.shape(), &copy, &tmp_copy, attr));
    *var->tensor() = *tmp_copy;
    var->tensor()->flat<T>().device(context->eigen_device<Device>()) =
        value.flat<T>();
    OP_REQUIRES_OK(context, CreateResource<Var>(
                                context, HandleFromInput(context, 0), var));
  }

 private:
  DataType dtype_;
};

// TODO(apassos) register for the GPU as well.
#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("CreateVariableOp")            \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          CreateVariableOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

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
    out->flat<T>().device(ctx->eigen_device<Device>()) =
        variable->tensor()->flat<T>();
  }
};

// TODO(apassos) register for the GPU as well.
#define REGISTER_KERNELS(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ReadVariableOp").Device(DEVICE_CPU).TypeConstraint<type>("dtype"), \
      ReadVariableOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class AssignVariableOp : public OpKernel {
 public:
  AssignVariableOp(OpKernelConstruction* c) : OpKernel(c) {}

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
    Tensor value = context->input(1);
    variable->tensor()->flat<T>().device(context->eigen_device<Device>()) =
        value.flat<T>();
  }
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

template <typename Device, typename T>
class AssignAddVariableOp : public OpKernel {
 public:
  AssignAddVariableOp(OpKernelConstruction* c) : OpKernel(c) {}

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
    Tensor value = context->input(1);
    variable->tensor()->flat<T>().device(context->eigen_device<Device>()) +=
        value.flat<T>();

    // TODO(apassos): this read can also be implemented efficiently so it is
    // free if no one uses the resulting tensor.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, variable->tensor()->shape(), &out));
    out->flat<T>().device(context->eigen_device<Device>()) =
        variable->tensor()->flat<T>();
  }
};

// TODO(apassos) register for the GPU as well.
#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("AssignAddVariableOp")         \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          AssignAddVariableOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp").Device(DEVICE_CPU),
                        IsResourceInitialized<Var>);

}  // namespace tensorflow
