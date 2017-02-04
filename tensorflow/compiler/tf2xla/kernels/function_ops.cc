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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

const char* const kGradientOp = "SymbolicGradient";

// Implementations of _ListToArray and _ArrayToList for functions.
class PassOn : public XlaOpKernel {
 public:
  explicit PassOn(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() == ctx->num_outputs(),
                errors::Internal("#inputs != #outputs : ", ctx->num_inputs(),
                                 " vs. ", ctx->num_outputs()));
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(
          ctx, input_type(i) == output_type(i),
          errors::Internal("Input and output types for position ", i,
                           " do not match: ", DataTypeString(input_type(i)),
                           " vs. ", DataTypeString(output_type(i))));
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      ctx->SetOutput(i, ctx->Input(i));
    }
  }
};

REGISTER_XLA_OP("_ListToArray", PassOn);
REGISTER_XLA_OP("_ArrayToList", PassOn);

// TODO(phawkins): this is an almost exact copy of the SymbolicGradientOp
// implementation from regular Tensorflow. Once XLA has been open sourced
// merge the two implementations. (Note: this implementation propagates the
// step_resource_manager).
class SymbolicGradientOp : public AsyncOpKernel {
 public:
  explicit SymbolicGradientOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), handle_(-1) {}

  ~SymbolicGradientOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);

    OP_REQUIRES_OK_ASYNC(
        ctx, lib->Instantiate(kGradientOp, def().attr(), &handle_), done);

    FunctionLibraryRuntime::Options opts;
    opts.step_id = ctx->step_id();
    opts.runner = ctx->runner();
    opts.step_container = ctx->step_container();
    std::vector<Tensor> args;
    args.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ctx->input(i));
    }
    std::vector<Tensor>* rets = new std::vector<Tensor>;
    lib->Run(
        opts, handle_, args, rets, [ctx, done, rets](const Status& status) {
          if (!status.ok()) {
            ctx->SetStatus(status);
          } else if (rets->size() != ctx->num_outputs()) {
            ctx->SetStatus(errors::InvalidArgument(
                "SymGrad expects to return ", ctx->num_outputs(),
                " tensor(s), but get ", rets->size(), " tensor(s) instead."));
          } else {
            for (size_t i = 0; i < rets->size(); ++i) {
              ctx->set_output(i, (*rets)[i]);
            }
          }
          delete rets;
          done();
        });
  }

 private:
  FunctionLibraryRuntime::Handle handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientOp);
};

REGISTER_XLA_OP(kGradientOp, SymbolicGradientOp);

}  // namespace
}  // namespace tensorflow
