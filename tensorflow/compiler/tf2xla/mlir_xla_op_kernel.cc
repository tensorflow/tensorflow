/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

namespace {

class MLIRContextResource : public ResourceBase {
 public:
  static constexpr const char* kDefaultResourceName =
      "mlir-xla-op-cached-context";

  static Status Create(MLIRContextResource** resource) {
    *resource = new MLIRContextResource();
    return absl::OkStatus();
  }
  mlir::MLIRContext* GetContext() { return &mlir_ctx_; }
  std::string DebugString() const override {
    return "MlirXlaOpKernel MLIRContext resource";
  }

 private:
  // Since this kernel implements lowering for a single TF operation, we
  // disable MLIR threading for efficiency purpose (avoid starting a large
  // number of threads eagerly).
  MLIRContextResource() : mlir_ctx_(mlir::MLIRContext::Threading::DISABLED) {}
  mlir::MLIRContext mlir_ctx_;
};

}  // namespace

Status MlirXlaOpKernel::ContextToXlaArgs(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>& xla_args) {
  // Collect arguments that are registered as CompileTimeConstantInput.
  std::vector<int> registered_consts_vec;
  TF_RETURN_IF_ERROR(tensorflow::XlaOpRegistry::CompileTimeConstantInputs(
      *this, &registered_consts_vec));
  llvm::SmallDenseSet<int, 4> registered_consts;
  registered_consts.insert(registered_consts_vec.begin(),
                           registered_consts_vec.end());

  int num_inputs = ctx->num_inputs();
  xla_args.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    // TODO(b/180448774): Handle kResource and kTensorList.
    XlaExpression::Kind ctx_kind_i = ctx->InputExpression(i).kind();
    if (ctx_kind_i != XlaExpression::Kind::kXlaOp &&
        ctx_kind_i != XlaExpression::Kind::kConstant)
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Input ", i, " to an MlirXlaOpKernel is invalid: ",
                       ctx->InputExpression(i).HumanString()));
    XlaCompiler::Argument arg;
    arg.type = ctx->input_type(i);
    arg.shape = ctx->InputXlaShape(i).value();
    arg.name = absl::StrCat("_arg", i);
    if (registered_consts.count(i)) {
      arg.kind = XlaCompiler::Argument::kConstant;
      TF_ASSIGN_OR_RETURN(arg.constant_value, ctx->ConstantInputTensor(i));
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
    }
    xla_args.push_back(arg);
  }
  return absl::OkStatus();
}

MlirXlaOpKernel::MlirXlaOpKernel(OpKernelConstruction* ctx)
    : XlaOpKernel(ctx) {}

Status MlirXlaOpKernel::ConstructXlaOp(XlaOpKernelContext* ctx) {
  // Create input XlaArguments.
  std::vector<XlaCompiler::Argument> xla_args;
  TF_RETURN_IF_ERROR(ContextToXlaArgs(ctx, xla_args));

  // Create input XlaOps.
  llvm::SmallVector<xla::XlaOp, 4> xla_params(ctx->num_inputs());
  for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
    xla_params[i] = ctx->Input(i);
  }

  // Create outputs.
  std::vector<DataType> result_dtypes(ctx->num_outputs());
  for (int i = 0, end = result_dtypes.size(); i < end; ++i) {
    result_dtypes[i] = ctx->expected_output_dtype(i);
  }

  // When there are no data-flow outputs from the node, the node is used as a
  // control output by the graph to TensorflowDialect importer.
  std::vector<std::string> control_rets;
  if (result_dtypes.empty()) {
    control_rets.push_back(def().name());
  }

  // Get the context's device.
  auto device = dynamic_cast<Device*>(ctx->op_kernel_context()->device());
  if (!device) {
    return tensorflow::errors::InvalidArgument(
        "Expected the XlaOpKernelContext argument's device to have type "
        "Device.");
  }

  // Create a graph that wraps the kernel.
  TF_ASSIGN_OR_RETURN(auto graph,
                      CreateSingleOpGraph(def(), xla_args, result_dtypes));

  ResourceMgr* res_manager = ctx->op_kernel_context()->resource_manager();
  MLIRContextResource* ctx_res;
  TF_RETURN_IF_ERROR(res_manager->LookupOrCreate<MLIRContextResource>(
      res_manager->default_container(),
      MLIRContextResource::kDefaultResourceName, &ctx_res,
      MLIRContextResource::Create));
  core::ScopedUnref unref_ctx(ctx_res);

  // Compile the graph to HLO.
  std::vector<xla::XlaOp> returns(1);
  auto build_hlo = [&](bool unconditionally_use_output_shapes) {
    return BuildHloFromGraph(
        *graph, *ctx->builder(), *ctx_res->GetContext(), xla_params, returns,
        unconditionally_use_output_shapes,
        mlir::SpanToArrayRef<XlaCompiler::Argument>(xla_args), control_rets,
        device->device_type(),
        *ctx->function_library()->GetFunctionLibraryDefinition());
  };

  // Some of the operations that come through here do not know how to set their
  // own output shapes (e.g. _XlaHostComputeMlir') so we may need to use the
  // unconditional output shapes option. However, many graphs fail if we do it
  // unconditionally so try both.
  if (!build_hlo(/*unconditionally_use_output_shapes=*/false).ok()) {
    // If that failed, then try again with the unconditional set true
    TF_RETURN_IF_ERROR(build_hlo(/*unconditionally_use_output_shapes=*/true));
  }

  // Set context outputs.
  for (int i = 0, end = returns.size(); i < end; ++i) {
    ctx->SetOutput(i, returns[i]);
  }

  return absl::OkStatus();
}

void MlirXlaOpKernel::Compile(XlaOpKernelContext* ctx) {
  auto status = ConstructXlaOp(ctx);
  if (!status.ok()) {
    errors::AppendToMessage(&status, "Failure to legalize ", def().name(),
                            " using MlirXlaOpKernel in the tf2xla bridge.");
  }
  OP_REQUIRES_OK(ctx, status);
}

}  // namespace tensorflow
