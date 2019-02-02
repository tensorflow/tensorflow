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

#include <memory>

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

// This file forms the basis of a stable ABI for third-party kernel
// implementations. It is crucial that changes to this file are made cautiously
// and with a focus on maintaining both source and binary compatibility.

struct TF_KernelBuilder {
  ::tensorflow::KernelDefBuilder* cc_builder;

  void* (*create_function)(TF_OpKernelConstruction*);
  void (*compute_function)(void*, TF_OpKernelContext*);
  void (*delete_function)(void*);
};

TF_KernelBuilder* TF_NewKernelBuilder(
    const char* op_name, const char* device_name,
    void* (*create_func)(TF_OpKernelConstruction*),
    void (*compute_func)(void*, TF_OpKernelContext*),
    void (*delete_func)(void*)) {
  TF_KernelBuilder* result = new TF_KernelBuilder;
  result->cc_builder = new ::tensorflow::KernelDefBuilder(op_name);
  result->cc_builder->Device(device_name);
  result->create_function = create_func;
  result->compute_function = compute_func;
  result->delete_function = delete_func;
  return result;
}

void TF_DeleteKernelBuilder(TF_KernelBuilder* builder) {
  if (builder != nullptr) {
    delete builder->cc_builder;
    delete builder;
  }
}

namespace tensorflow {
namespace {

// An OpKernel whose methods delegate to C function pointers.
class COpKernel : public OpKernel {
 public:
  explicit COpKernel(OpKernelConstruction* ctx,
                     void* (*create_func)(TF_OpKernelConstruction*),
                     void (*compute_func)(void*, TF_OpKernelContext*),
                     void (*delete_func)(void*))
      : OpKernel(ctx), compute_func_(compute_func), delete_func_(delete_func) {
    if (create_func != nullptr) {
      c_kernel_ =
          (*create_func)(reinterpret_cast<TF_OpKernelConstruction*>(ctx));
    } else {
      c_kernel_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    (*compute_func_)(c_kernel_, reinterpret_cast<TF_OpKernelContext*>(ctx));
  }

  ~COpKernel() override {
    if (delete_func_ != nullptr) {
      (*delete_func_)(c_kernel_);
    }
  }

 private:
  void (*compute_func_)(void*, TF_OpKernelContext* context);
  void (*delete_func_)(void*);
  void* c_kernel_;
};

// A KernelFactory that returns COpKernel instances.
class KernelBuilderFactory
    : public ::tensorflow::kernel_factory::OpKernelFactory {
 public:
  explicit KernelBuilderFactory(TF_KernelBuilder* builder)
      : builder_(builder) {}
  ::tensorflow::OpKernel* Create(
      ::tensorflow::OpKernelConstruction* context) override {
    return new ::tensorflow::COpKernel(context, builder_->create_function,
                                       builder_->compute_function,
                                       builder_->delete_function);
  }
  ~KernelBuilderFactory() override { TF_DeleteKernelBuilder(builder_); }

 private:
  TF_KernelBuilder* builder_;
};
}  // namespace
}  // namespace tensorflow

void TF_RegisterKernelBuilder(const char* name, TF_KernelBuilder* builder,
                              TF_Status* status) {
  using tensorflow::register_kernel::Name;

  tensorflow::kernel_factory::OpKernelRegistrar(
      builder->cc_builder->Build(), name,
      absl::make_unique<tensorflow::KernelBuilderFactory>(builder));

  TF_SetStatus(status, TF_OK, "");
}

int TF_NumInputs(TF_OpKernelContext* ctx) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return cc_ctx->num_inputs();
}

int TF_NumOutputs(TF_OpKernelContext* ctx) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return cc_ctx->num_outputs();
}

void TF_GetInput(TF_OpKernelContext* ctx, int i, TF_Tensor** tensor,
                 TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }
  const ::tensorflow::Tensor& cc_tensor(cc_ctx->input(i));
  TF_Tensor* result = ::tensorflow::TF_TensorFromTensor(cc_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    *tensor = result;
  }
}

void TF_SetOutput(TF_OpKernelContext* ctx, int i, const TF_Tensor* tensor,
                  TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }
  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(status, s);
  if (s.ok()) {
    cc_ctx->set_output(i, cc_tensor);
  }
}

void TF_OpKernelConstruction_Failure(TF_OpKernelConstruction* ctx,
                                     TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s(::tensorflow::StatusFromTF_Status(status));
  cc_ctx->CtxFailure(s);
}

void TF_OpKernelContext_Failure(TF_OpKernelContext* ctx, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  ::tensorflow::Status s(::tensorflow::StatusFromTF_Status(status));
  cc_ctx->CtxFailure(s);
}

#define DEFINE_TF_GETATTR_(struct_name, func, c_type, cc_type)                 \
  void struct_name##_GetAttr##func(struct_name* ctx, const char* attr_name,    \
                                   c_type* val, TF_Status* status) {           \
    TF_SetStatus(status, TF_OK, "");                                           \
    cc_type v;                                                                 \
    auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx); \
    ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);                   \
    ::tensorflow::Set_TF_Status_from_Status(status, s);                        \
    if (s.ok()) {                                                              \
      *val = static_cast<c_type>(v);                                           \
    }                                                                          \
  }

#define DEFINE_TF_GETATTR(func, c_type, cc_type)                     \
  DEFINE_TF_GETATTR_(TF_OpKernelConstruction, func, c_type, cc_type) \
  DEFINE_TF_GETATTR_(TF_OpKernelContext, func, c_type, cc_type)

DEFINE_TF_GETATTR(Type, TF_DataType, tensorflow::DataType)

TF_DataType TF_ExpectedOutputDataType(TF_OpKernelContext* ctx, int i) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return static_cast<TF_DataType>(cc_ctx->expected_output_dtype(i));
}

int64_t TF_StepId(TF_OpKernelContext* ctx) {
  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)->step_id();
}
