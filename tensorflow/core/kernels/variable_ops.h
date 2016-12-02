/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_VARIABLE_OPS_H_
#define TENSORFLOW_KERNELS_VARIABLE_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource stored by variables in the resource manager.
class Var : public ResourceBase {
 public:
  explicit Var(DataType dtype) : tensor_(dtype) {}
  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tensor_;

  ~Var() override {}
  TF_DISALLOW_COPY_AND_ASSIGN(Var);
};

class VariableOp : public OpKernel {
 public:
  explicit VariableOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    dtype_ = RemoveRefType(context->output_type(0));
  }

  ~VariableOp() override {
    if (var_) var_->Unref();
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(init_mu_);
    if (var_ == nullptr) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      true /* use name() */));
      auto creator = [this](Var** var) {
        *var = new Var(dtype_);
        (*var)->tensor()->set_shape(shape_);
        return Status::OK();
      };
      OP_REQUIRES_OK(ctx,
                     cinfo_.resource_manager()->LookupOrCreate<Var>(
                         cinfo_.container(), cinfo_.name(), &var_, creator));
    }
    // Output a reference to our tensor, so it may be updated.
    //
    // As long as *this is alive, the ref we return here is valid
    // because *this owns a ref on var_.
    ctx->set_output_ref(0, var_->mu(), var_->tensor());
  }

 private:
  DataType dtype_;
  TensorShape shape_;

  mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  Var* var_ GUARDED_BY(init_mu_) = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(VariableOp);
};

class TemporaryVariableOp : public OpKernel {
 public:
  explicit TemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    // Variable name defaults to op name if not specified explicitly.
    if (var_name_ == "") var_name_ = name();
  }

  void Compute(OpKernelContext* context) override {
    Status s;
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    auto* tmp_var = new TmpVar;
    OP_REQUIRES(context, tmp_var,
                errors::ResourceExhausted("Could not allocate TmpVar."));
    tmp_var->name = var_name_;
    s = context->allocate_temp(dtype_, shape_, &tmp_var->val);
    if (!s.ok()) tmp_var->Unref();
    OP_REQUIRES_OK(context, s);
    OP_REQUIRES_OK(context, rm->Create(context->step_container()->name(),
                                       var_name_, tmp_var));
    context->set_output_ref(0, &tmp_var->mu, &tmp_var->val);
  }

 private:
  // Refcounted temporary variable resource.
  friend class DestroyTemporaryVariableOp;
  struct TmpVar : public ResourceBase {
    mutex mu;
    Tensor val;
    string name;
    string DebugString() override { return name; }
    ~TmpVar() override { VLOG(3) << "TmpVar " << name << " deleted"; }
  };

  TensorShape shape_;
  DataType dtype_;
  string var_name_;
};

class DestroyTemporaryVariableOp : public OpKernel {
 public:
  explicit DestroyTemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"))
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    OP_REQUIRES(context, var_name_ != "",
                errors::InvalidArgument("Missing var_name attribute"));
  }

  void Compute(OpKernelContext* context) override {
    // NOTE(pbar): All other mutators of the Tensor Ref *must* have completed
    // their execution before this DestroyTemporaryVariable op executes.
    // This is typically achieved using control dependencies.
    CHECK(IsRefType(context->input_dtype(0)));
    Tensor tmpvar = context->mutable_input(0, false);
    context->set_output(0, tmpvar);
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    OP_REQUIRES_OK(context, rm->Delete<TemporaryVariableOp::TmpVar>(
                                context->step_container()->name(), var_name_));
  }

 private:
  string var_name_;
};

class IsVariableInitializedOp : public OpKernel {
 public:
  IsVariableInitializedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get a mutable input tensor of the Ref input.
    const Tensor& input_tensor = context->mutable_input(0, false);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    bool result = input_tensor.IsInitialized();
    output_tensor() = result;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_VARIABLE_OPS_H_
