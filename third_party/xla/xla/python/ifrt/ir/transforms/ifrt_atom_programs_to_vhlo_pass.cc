/* Copyright 2024 The OpenXLA Authors.

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
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace ifrt {

namespace {

class IfrtAtomProgramsToVhloPass
    : public mlir::PassWrapper<IfrtAtomProgramsToVhloPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit IfrtAtomProgramsToVhloPass(
      tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>* atom_programs,
      std::string vhlo_target_version)
      : atom_programs_(atom_programs),
        vhlo_target_version_(std::move(vhlo_target_version)) {}

  llvm::StringRef getArgument() const override {
    return "ifrt-atom-programs-to-vhlo";
  }

  llvm::StringRef getDescription() const override {
    return "Populates a map from unique atom program name VHLO bytecode.";
  }

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    mlir::registerAllDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
  }

  void runOnOperation() override;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfrtAtomProgramsToVhloPass);

 private:
  tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>* atom_programs_;
  std::string vhlo_target_version_;
};

void IfrtAtomProgramsToVhloPass::runOnOperation() {
  mlir::SymbolTableCollection symbol_table;
  mlir::MLIRContext& context = getContext();
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = getOperation();

  // Create a new context and register the dialects that are loaded in the
  // current context. This context will be used to temporarily clone atom
  // programs into, and run the to VHLO conversion passes. It is necessary to
  // do this because these passes change all the types in the context.
  mlir::MLIRContext tmp_context;
  mlir::OpBuilder tmp_builder(&tmp_context);
  // Keeps track of the atom programs that have already been serialized.
  absl::flat_hash_set<std::string> converted_atom_program_names;

  // Walk the module and convert each atom program to VHLO.
  auto result = module.walk([&](CallOp call_op) -> mlir::WalkResult {
    mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
    if (callee == nullptr) {
      return call_op->emitOpError()
             << "can't find callee `" << call_op.getCalleeAttr() << "`";
    }
    auto stablehlo_module = llvm::cast<mlir::ModuleOp>(callee->getParentOp());
    if (stablehlo_module == module) {
      return call_op->emitOpError() << "callee `" << call_op.getCalleeAttr()
                                    << "` has not been outlined to a module";
    }
    // Verify that the atom program is a top-level IFRT IR module. Nested atom
    // programs are not supported in IFRT IR. Moreover, it would difficult
    // to exactly reconstruct the IFRT IR program post atom program DCE.
    // For example, in the example below, DCE would remove the entire module
    // @outer, which would not be possible to reconstruct on deserialization.
    // module @outer {
    //  module @atom_program1 {}
    //  module @atom_program2 {}
    // }
    if (call_op.getCalleeAttr().getNestedReferences().size() > 1) {
      return call_op->emitOpError() << "nested atom programs are not supported "
                                    << call_op.getCalleeAttr();
    }
    if (!stablehlo_module.getSymNameAttr()) {
      return call_op->emitOpError()
             << "callee `" << call_op.getCalleeAttr()
             << "` has not been outlined to a module with a `sym_name`";
    }
    std::string atom_program_name = stablehlo_module.getSymNameAttr().str();
    if (!converted_atom_program_names.insert(atom_program_name).second) {
      // Skip if the atom program has already been serialized.
      return mlir::WalkResult::advance();
    }

    // Clone the module into a tmp context.
    auto tmp_module = CloneModuleUsingBuilder(stablehlo_module, tmp_builder);
    absl::Cleanup erase_tmp_module = [&]() { tmp_module.erase(); };
    // Convert the tmp module as VHLO.
    IfrtIrAtomProgramProto* atom_program_proto = atom_programs_->Add();
    atom_program_proto->set_name(atom_program_name);
    atom_program_proto->set_version(vhlo_target_version_);
    llvm::raw_string_ostream os(*atom_program_proto->mutable_program());
    if (mlir::failed(mlir::stablehlo::serializePortableArtifact(
            tmp_module, vhlo_target_version_, os))) {
      return stablehlo_module->emitOpError() << "failed to serialize to VHLO";
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtAtomProgramsToVhloPass(
    tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>* atom_programs,
    std::string vhlo_target_version) {
  return std::make_unique<IfrtAtomProgramsToVhloPass>(
      atom_programs, std::move(vhlo_target_version));
}

}  // namespace ifrt
}  // namespace xla
