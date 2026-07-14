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

#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTDUMPATOMPROGRAMSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

absl::Status DumpOperation(mlir::Operation* op, std::string dump_dir,
                           std::string filename) {
  std::string file_path =
      tsl::io::JoinPath(dump_dir, absl::StrCat(filename, ".mlir"));
  return tsl::WriteStringToFile(tsl::Env::Default(), file_path,
                                OperationToString(op, mlir::OpPrintingFlags()));
}

// Empties the body of `func_op` and replaces it with a return op.
void EmptyFunctionBody(mlir::func::FuncOp func_op, mlir::OpBuilder& builder) {
  func_op.eraseBody();
  mlir::Block* entry_block = func_op.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(entry_block);
  llvm::SmallVector<mlir::Value> dummy_res;
  for (mlir::Type res_type : func_op.getFunctionType().getResults()) {
    dummy_res.push_back(
        mlir::ub::PoisonOp::create(builder, func_op.getLoc(), res_type));
  }
  mlir::func::ReturnOp::create(builder, func_op.getLoc(), dummy_res);
}

class IfrtDumpAtomProgramsPass
    : public impl::IfrtDumpAtomProgramsPassBase<IfrtDumpAtomProgramsPass> {
 public:
  using impl::IfrtDumpAtomProgramsPassBase<
      IfrtDumpAtomProgramsPass>::IfrtDumpAtomProgramsPassBase;

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::ub::UBDialect>();
  }

  void runOnOperation() override {
    if (dump_dir.empty()) {
      return signalPassFailure();
    }

    mlir::SymbolTableCollection symbol_table;
    // Keeps track of the atom programs that have already been dumped.
    absl::flat_hash_set<std::string> dumped_atom_program_names;

    mlir::func::FuncOp main_func = GetMainFunction(getOperation());
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
              call_op->emitOpError()
                  << "failed to dump atom program: " << status.ToString();
              return mlir::WalkResult::interrupt();
            }
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }

    // Clones the module because the bodies of non-IFRT functions are erased
    // to keep the IFRT IR dump concise, but valid.
    mlir::ModuleOp module_op = getOperation().clone();
    absl::Cleanup cleanup([&module_op] { module_op.erase(); });
    mlir::OpBuilder builder(module_op);
    llvm::SmallVector<mlir::func::FuncOp, 128> funcs_to_empty;
    module_op.walk([&](mlir::func::FuncOp func_op) -> mlir::WalkResult {
      if (!IsIfrtFunction(func_op)) {
        funcs_to_empty.push_back(func_op);
      }
      return mlir::WalkResult::advance();
    });
    for (mlir::func::FuncOp func_op : funcs_to_empty) {
      EmptyFunctionBody(func_op, builder);
    }
    if (auto status = DumpOperation(module_op, dump_dir, "ifrt_only");
        !status.ok()) {
      main_func->emitOpError()
          << "failed to dump main IFRT module: " << status.ToString();
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla
