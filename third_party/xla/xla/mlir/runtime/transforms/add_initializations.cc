/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_dialect.h"
#include "xla/mlir/runtime/transforms/passes.h"
#include "xla/mlir/runtime/utils/custom_calls.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_ADDINITIALIZATIONS
#include "xla/mlir/runtime/transforms/passes.h.inc"

class AddInitializations
    : public impl::AddInitializationsBase<AddInitializations> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------====/

void AddInitializations::runOnOperation() {
  ModuleOp module = getOperation();
  bool requires_blas = false;
  if (Attribute requires_blas_attr = module->getAttr(kRequiresBlasAttrName)) {
    requires_blas = cast<BoolAttr>(requires_blas_attr).getValue();
  }

  if (!requires_blas) {
    return;
  }

  SymbolTable sym_table(module);
  CustomCallDeclarations custom_calls(std::move(sym_table));

  ImplicitLocOpBuilder b(module->getLoc(), custom_calls.sym_table().getOp());
  func::FuncOp initialize_cublas = custom_calls.GetOrCreate(
      b, "xla.gpu.init_cublas", TypeRange(), TypeRange());

  module.walk([&](func::FuncOp func) {
    if (IntegerAttr exported = func.getOperation()->getAttrOfType<IntegerAttr>(
            kExportedAttrName)) {
      int64_t ordinal = exported.getInt();
      if (ordinal == 0) {
        b.setInsertionPointToStart(&*func.getBody().getBlocks().begin());
        b.create<func::CallOp>(initialize_cublas.getName(), TypeRange());
      }
    }
  });
}

std::unique_ptr<OperationPass<ModuleOp>> CreateAddInitializationsPass() {
  return std::make_unique<AddInitializations>();
}

}  // namespace runtime
}  // namespace xla
