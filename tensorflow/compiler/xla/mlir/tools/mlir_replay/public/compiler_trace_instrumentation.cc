/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/compiler_trace_instrumentation.h"

#include <string>

#include "absl/strings/str_format.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/path.h"

namespace mlir {
namespace interpreter {

void MlirCompilerTraceInstrumentation::runAfterPass(Pass* pass, Operation* op) {
  ModuleOp module = llvm::dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<mlir::ModuleOp>();
  }
  if (!module) {
    LOG(ERROR) << "Failed to find a ModuleOp: " << pass->getName().str() << ".";
    return;
  }

  auto* item = trace_.mutable_passes()->Add();
  item->set_after_pass(pass->getName().str());
  llvm::raw_string_ostream os(*item->mutable_mlir_module());
  module.print(os);
}

MlirCompilerTraceInstrumentation::~MlirCompilerTraceInstrumentation() {
  if (!trace_.passes().empty()) {
    std::string filename;
    absl::StrAppendFormat(&filename, "module_%04d", unique_id_);
    if (!module_name_.empty()) {
      absl::StrAppend(&filename, ".", module_name_);
    }
    absl::StrAppend(&filename, ".mlir-trace.pb");
    filename = tsl::io::JoinPath(dirname_, filename);
    TF_CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(dirname_));
    TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), filename, trace_));
  }
}

}  // namespace interpreter
}  // namespace mlir
