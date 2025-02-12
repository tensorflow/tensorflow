/* Copyright 2025 The OpenXLA Authors.

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
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTDUMPATOMPROGRAMSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

absl::Status DumpOperation(mlir::Operation* op, std::string dump_dir,
                           std::string filename) {
  std::string file_path =
      tsl::io::JoinPath(dump_dir, absl::StrCat(filename, ".mlir"));
  return tsl::WriteStringToFile(tsl::Env::Default(), file_path,
                                OperationToString(op, mlir::OpPrintingFlags()));
}

class IfrtDumpAtomProgramsPass
    : public impl::IfrtDumpAtomProgramsPassBase<IfrtDumpAtomProgramsPass> {
 public:
  using impl::IfrtDumpAtomProgramsPassBase<
      IfrtDumpAtomProgramsPass>::IfrtDumpAtomProgramsPassBase;

  void runOnOperation() override {
    if (dump_dir.empty()) {
      return signalPassFailure();
    }

    mlir::SymbolTableCollection symbol_table;
    mlir::ModuleOp module_op = getOperation();
    // Keeps track of the atom programs that have already been dumped.
    absl::flat_hash_set<std::string> dumped_atom_program_names;

    auto main_func = GetMainFunction(module_op);

    // Clones the main function to ensure that the attribute aliases are
    // preserved while printing. Otherwise, the op would be printed in its
    // full form (i.e., every argument with the entire device list expanded)
    // and would lead to large ifrt dump files.
    auto cloned_main = main_func.clone();
    if (auto status = DumpOperation(cloned_main, dump_dir, "ifrt_main_func");
        !status.ok()) {
      cloned_main.erase();
      main_func->emitOpError()
          << "failed to dump main func: " << status.ToString();
      signalPassFailure();
      return;
    }
    cloned_main.erase();

    mlir::WalkResult result =
        main_func.walk([&](CallOp call_op) -> mlir::WalkResult {
          mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
          CHECK(callee != nullptr);
          auto atom_program_module =
              llvm::cast<mlir::ModuleOp>(callee->getParentOp());
          std::string atom_program_name =
              atom_program_module.getSymNameAttr().str();
          if (dumped_atom_program_names.insert(atom_program_name).second) {
            if (auto status = DumpOperation(atom_program_module, dump_dir,
                                            atom_program_name);
                !status.ok()) {
              return call_op->emitOpError()
                     << "failed to dump atom program: " << status.ToString();
            }
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtDumpAtomProgramsPass(IfrtDumpAtomProgramsPassOptions options) {
  return std::make_unique<IfrtDumpAtomProgramsPass>(options);
}

}  // namespace ifrt
}  // namespace xla
