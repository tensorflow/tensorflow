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
#include <utility>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace mlir_gpu {

void MlirIrGenTestBase::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const string& pattern,
    LoweringStage stage) {
  MlirCompiler* compiler = GetMLIRCompiler();
  string ir;
  compiler->SetModuleHook({[&ir](mlir::ModuleOp module) -> Status {
                             std::string buffer_string;
                             llvm::raw_string_ostream ostream(buffer_string);
                             module.print(ostream);
                             ostream.flush();
                             ir = buffer_string;
                             return Status::OK();
                           },
                           stage});
  Status status = CompileToExecutable(std::move(hlo_module)).status();
  compiler->RemoveModuleHook();
  TF_ASSERT_OK(status);

  StatusOr<bool> filecheck_result = RunFileCheck(ir, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

void MlirIrGenTestBase::CompileAndVerifyIr(const string& hlo_text,
                                           const string& expected_llvm_ir,
                                           LoweringStage stage) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_text, config));
  CompileAndVerifyIr(std::move(module), expected_llvm_ir, stage);
}

MlirCompiler* MlirIrGenTestBase::GetMLIRCompiler() {
  // TODO(b/137624192): Remove failover once no longer in place.
  auto* failover = static_cast<FailoverCompiler*>(backend().compiler());
  return static_cast<MlirCompiler*>(failover->GetPrimary());
}

}  // namespace mlir_gpu
}  // namespace xla
