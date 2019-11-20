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

#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_irgen_test_base.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/inject_errors_pass.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace mlir_gpu {

void MlirIrGenTestBase::CompileIr(std::unique_ptr<HloModule> hlo_module,
                                  const MlirCompiler::IRHook& ir_hook) {
  MlirCompiler* compiler = GetMLIRCompiler();
  compiler->SetModuleHook(ir_hook);
  Status status = CompileToExecutable(std::move(hlo_module)).status();
  compiler->RemoveModuleHook();
  TF_ASSERT_OK(status);
}

void MlirIrGenTestBase::PatternMatch(const string& str, const string& pattern) {
  StatusOr<bool> filecheck_result = RunFileCheck(str, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

string MlirIrGenTestBase::CompileIr(
    std::unique_ptr<HloModule> hlo_module,
    MlirCompiler::IRHook::LoweringStage printing_stage) {
  string ir;
  CompileIr(std::move(hlo_module),
            {[&ir](mlir::ModuleOp module) -> Status {
               std::string buffer_string;
               llvm::raw_string_ostream ostream(buffer_string);
               module.print(ostream);
               ostream.flush();
               ir = buffer_string;
               return Status::OK();
             },
             printing_stage});
  return ir;
}

void MlirIrGenTestBase::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const string& pattern,
    LoweringStage printing_stage) {
  string ir = CompileIr(std::move(hlo_module), printing_stage);
  PatternMatch(ir, pattern);
}

void MlirIrGenTestBase::CompileAndVerifyIr(const string& hlo_text,
                                           const string& expected_llvm_ir,
                                           LoweringStage printing_stage) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  auto module = absl::make_unique<VerifiedHloModule>(
      "Module", config, /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      /*shape_size_function=*/ShapeUtil::ByteSizeOfElements);
  TF_ASSERT_OK(module->ParseHloStringAndVerifyModule(hlo_text));
  CompileAndVerifyIr(std::move(module), expected_llvm_ir, printing_stage);
}

MlirCompiler::IRHook MlirIrGenTestBase::getIRHookBreakingLoweringStage(
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

StatusOr<string> MlirIrGenTestBase::CompileAndInjectErrors(
    std::unique_ptr<HloModule> hlo_module, LoweringStage breaking_stage) {
  string errors;
  auto error_handler = [&errors](const EmissionContext::ErrorMap& error_map,
                                 HloModule* hlo_module) {
    errors = "ERRORS FOUND: ";
    for (auto& err : error_map) {
      errors += "[" + err.first->ToString() + ": " +
                absl::StrJoin(err.second, "; ") + "]";
    }
  };

  MlirCompiler* compiler = GetMLIRCompiler();
  compiler->SetModuleHook(getIRHookBreakingLoweringStage(breaking_stage));
  compiler->SetErrorHandler(error_handler);
  Status status = CompileToExecutable(std::move(hlo_module)).status();
  compiler->RemoveModuleHook();
  compiler->RemoveErrorHandler();

  if (status.ok()) {
    return errors;
  }
  return status;
}

void MlirIrGenTestBase::CompileAndVerifyErrors(const string& hlo_text,
                                               const string& expected_errors,
                                               LoweringStage breaking_stage) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  auto module = absl::make_unique<VerifiedHloModule>(
      "Module", config, /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      /*shape_size_function=*/ShapeUtil::ByteSizeOfElements);
  TF_ASSERT_OK(module->ParseHloStringAndVerifyModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      string errors, CompileAndInjectErrors(std::move(module), breaking_stage));
  PatternMatch(errors, expected_errors);
}

MlirCompiler* MlirIrGenTestBase::GetMLIRCompiler() {
  // TODO(b/137624192): Remove failover once no longer in place.
  auto* failover = static_cast<FailoverCompiler*>(backend().compiler());
  return static_cast<MlirCompiler*>(failover->GetPrimary());
}

}  // namespace mlir_gpu
}  // namespace xla
