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

#include "tensorflow/core/function/runtime_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace core {
namespace function {

EagerContext& GlobalEagerContext() {
  static EagerContext* global_ctx = []() {
    SessionOptions opts;
    std::vector<std::unique_ptr<Device>> devices;
    Status&& device_init_status = DeviceFactory::AddDevices(
        opts, "/job:localhost/replica:0/task:0", &devices);
    CHECK(device_init_status.ok());  // Crash OK

    return new EagerContext(
        opts, ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /*async=*/false,
        /*device_mgr=*/new DynamicDeviceMgr(std::move(devices)),
        /*device_mgr_owned=*/true,
        /*rendezvous=*/nullptr,
        /*cluster_flr=*/nullptr,
        /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/false);
  }();
  return *global_ctx;
}

EagerContext& GlobalPythonEagerContext() {
  EagerContext* ctx = reinterpret_cast<EagerContext*>(GetCEagerContext());
  DCHECK(ctx) << "The Python eager context must be initialized first.";
  return *ctx;
}

StatusOr<FunctionDef> Runtime::GetFunctionProto(StringPiece name) {
  EagerContext& ctx = this->eager_ctx_;

  const FunctionDef* f = ctx.FindFunctionDef(std::string(name));
  if (f == nullptr) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Could not find an attribute for key ", name));
  }

  return *f;
}

Status Runtime::CreateFunction(const FunctionDef& fdef) {
  const auto& fname = fdef.signature().name();
  if (this->eager_ctx_.FindFunctionByName(fname)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(this->eager_ctx_.RemoveFunction(fname),
                                    "removing function ", fname);
  }
  return this->eager_ctx_.AddFunctionDef(fdef);
}

Status Runtime::CreateFunction(OpaqueTfgGraphFuncOp* fop) {
  mlir::tfg::GraphFuncOp fop_proper =
      *reinterpret_cast<mlir::tfg::GraphFuncOp*>(fop);
  return mlir::tfg::ConvertToFunctionDef(fop_proper,
                                         *this->eager_ctx_.FuncLibDef());
}

Status Runtime::TransformFunction(StringPiece name, StringPiece pipeline_name) {
  // TODO(mdan): Use a longer-lived context.
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  std::string error;
  llvm::raw_string_ostream error_stream(error);
  // StringPiece doesn't seem to always be compatible with StringRef.
  if (mlir::failed(mlir::parsePassPipeline(std::string(pipeline_name), pm,
                                           error_stream))) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("locating pass pipeline ", pipeline_name, ": ",
                               error_stream.str()));
  }

  // For now, we roundtrip from proto. Once we have a permanent MLIR
  // representation, we should be able to use it directly.
  auto fn = GetFunctionProto(name);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(fn.status(), "loading function ", name);

  GraphDef graph;
  *graph.mutable_library()->add_function() = *fn;
  tensorflow::GraphDebugInfo debug_info;
  auto mlir_fn = mlir::tfg::ImportGraphDef(&ctx, debug_info, graph);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(mlir_fn.status(), "importing function ",
                                  name);

  mlir::StatusScopedDiagnosticHandler diagnostics_handler(&ctx);
  if (failed(pm.run(mlir_fn->get()))) {
    return diagnostics_handler.Combine(
        Status(error::INVALID_ARGUMENT,
               absl::StrCat("running pass pipeline ", pipeline_name, ": ")));
  }

  for (auto fn : mlir_fn->get().getBody()->getOps<mlir::tfg::GraphFuncOp>()) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        CreateFunction(reinterpret_cast<OpaqueTfgGraphFuncOp*>(&fn)),
        absl::StrCat("updating function ", fn.getName().str()));
  }

  return OkStatus();
}

StatusOr<ReturnValues> Runtime::CallFunction(
    StringPiece name, absl::Span<AbstractTensorHandle* const> args) {
  EagerContext& ctx = this->eager_ctx_;

  ImmediateOpPtr op(ctx.CreateOperation());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(op->Reset(name.data(), nullptr),
                                  "initializing call op for ", name);

  TF_RETURN_WITH_CONTEXT_IF_ERROR(op->AddInputList(args),
                                  "preparing call args for ", name);

  const FunctionDef* fn_def = ctx.GetFunctionDef(string(name));
  int num_retvals = fn_def->signature().output_arg_size();
  int actual_retvals = num_retvals;
  std::vector<ImmediateExecutionTensorHandle*> retvals(num_retvals);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      op->Execute(absl::MakeSpan(
                      reinterpret_cast<AbstractTensorHandle**>(retvals.data()),
                      num_retvals),
                  &actual_retvals),
      "executing call op for ", name);
  DCHECK(num_retvals == actual_retvals);

  ReturnValues final_returns;
  for (const auto& r : retvals) {
    final_returns.emplace_back(ImmediateTensorHandlePtr(r));
  }

  return final_returns;
}

}  // namespace function
}  // namespace core
}  // namespace tensorflow
