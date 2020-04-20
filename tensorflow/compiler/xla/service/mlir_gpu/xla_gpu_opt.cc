/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/xla_gpu_opt.h"

#include <memory>
#include <string>

#include "absl/strings/str_join.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/inject_errors_pass.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace mlir_gpu {

Status XlaGpuOpt::CompileIr(std::unique_ptr<HloModule> hlo_module,
                            const MlirCompiler::IRHook& ir_hook) {
  MlirCompiler* compiler = GetMLIRCompiler();
  compiler->SetModuleHook(ir_hook);
  TF_ASSIGN_OR_RETURN(hlo_module, backend_->compiler()->RunHloPasses(
                                      std::move(hlo_module),
                                      backend_->default_stream_executor(),
                                      /*device_allocator=*/nullptr));
  Status status = backend_->compiler()
                      ->RunBackend(std::move(hlo_module),
                                   backend_->default_stream_executor(),
                                   /*device_allocator=*/nullptr)
                      .status();
  compiler->RemoveModuleHook();
  return status;
}

StatusOr<std::string> XlaGpuOpt::CompileIr(
    std::unique_ptr<HloModule> hlo_module,
    MlirCompiler::IRHook::LoweringStage printing_stage) {
  std::string ir;
  TF_RETURN_IF_ERROR(CompileIr(
      std::move(hlo_module), {[&ir](mlir::ModuleOp module) -> Status {
                                std::string buffer_string;
                                llvm::raw_string_ostream ostream(buffer_string);
                                module.print(ostream);
                                ostream.flush();
                                ir = buffer_string;
                                return Status::OK();
                              },
                              printing_stage}));
  return ir;
}

Status XlaGpuOpt::CompileAndOutputIr(std::unique_ptr<HloModule> hlo_module,
                                     llvm::raw_ostream& os,
                                     LoweringStage printing_stage) {
  TF_ASSIGN_OR_RETURN(std::string ir,
                      CompileIr(std::move(hlo_module), printing_stage));
  os << ir;
  return Status::OK();
}

Status XlaGpuOpt::CompileAndOutputIr(const std::string& hlo_text,
                                     llvm::raw_ostream& os,
                                     LoweringStage printing_stage) {
  TF_ASSIGN_OR_RETURN(auto module, GetVerifiedHloModule(hlo_text));
  return CompileAndOutputIr(std::move(module), os, printing_stage);
}

MlirCompiler::IRHook XlaGpuOpt::GetIRHookBreakingLoweringStage(
    LoweringStage breaking_stage) {
  return {[](mlir::ModuleOp module) -> Status {
            mlir::PassManager pm(module.getContext());
            pm.addPass(::mlir::createInjectErrorsForTestingPass());
            if (failed(pm.run(module))) {
              return InternalError("InjectErrorsForTestingPass failed.");
            }
            return Status::OK();
          },
          breaking_stage};
}

StatusOr<string> XlaGpuOpt::CompileAndInjectErrors(
    std::unique_ptr<HloModule> hlo_module, LoweringStage breaking_stage) {
  std::string errors;
  auto error_handler = [&errors](const EmissionContext::ErrorMap& error_map,
                                 HloModule* hlo_module) {
    errors = "ERRORS FOUND: ";
    for (auto& err : error_map) {
      errors += "[" + err.first->ToString() + ": " +
                absl::StrJoin(err.second, "; ") + "]";
    }
  };

  MlirCompiler* compiler = GetMLIRCompiler();
  compiler->SetModuleHook(GetIRHookBreakingLoweringStage(breaking_stage));
  compiler->SetErrorHandler(error_handler);
  TF_ASSIGN_OR_RETURN(
      hlo_module, compiler->RunHloPasses(std::move(hlo_module),
                                         backend_->default_stream_executor(),
                                         /*device_allocator=*/nullptr));
  Status status = compiler
                      ->RunBackend(std::move(hlo_module),
                                   backend_->default_stream_executor(),
                                   /*device_allocator=*/nullptr)
                      .status();
  compiler->RemoveModuleHook();
  compiler->RemoveErrorHandler();
  if (status.ok()) {
    return errors;
  }
  return status;
}

Status XlaGpuOpt::CompileAndExpectErrors(const std::string& hlo_text,
                                         llvm::raw_ostream& os,
                                         LoweringStage breaking_stage) {
  TF_ASSIGN_OR_RETURN(auto module, GetVerifiedHloModule(hlo_text));
  TF_ASSIGN_OR_RETURN(
      std::string errors,
      CompileAndInjectErrors(std::move(module), breaking_stage));
  os << errors;
  return Status::OK();
}

StatusOr<std::unique_ptr<VerifiedHloModule>> XlaGpuOpt::GetVerifiedHloModule(
    const std::string& hlo_text) {
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsFromFlags();
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  config.set_debug_options(debug_options);
  auto module = absl::make_unique<VerifiedHloModule>(
      "Module", config, /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      /*shape_size_function=*/ShapeUtil::ByteSizeOfElements);
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  return std::move(module);
}

MlirCompiler* XlaGpuOpt::GetMLIRCompiler() {
  // TODO(b/137624192): Remove failover once no longer in place.
  auto* failover = static_cast<FailoverCompiler*>(backend_->compiler());
  return static_cast<MlirCompiler*>(failover->GetPrimary());
}

}  // namespace mlir_gpu
}  // namespace xla
