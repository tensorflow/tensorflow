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

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace ifrt {

namespace {

class IfrtAtomProgramsFromVhloPass
    : public mlir::PassWrapper<IfrtAtomProgramsFromVhloPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit IfrtAtomProgramsFromVhloPass(
      const tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>&
          atom_programs)
      : atom_programs_(atom_programs) {}

  llvm::StringRef getArgument() const override {
    return "ifrt-atom-programs-from-vhlo";
  }

  llvm::StringRef getDescription() const override {
    return "Converts atom programs from VHLO and adds them to the module.";
  }

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    mlir::registerAllDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
  }

  void runOnOperation() override;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfrtAtomProgramsFromVhloPass);

 private:
  const tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>& atom_programs_;
};

void IfrtAtomProgramsFromVhloPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext& context = getContext();
  mlir::OpBuilder builder(&context);
  for (const auto& atom_program_proto : atom_programs_) {
    auto atom_program_module = mlir::stablehlo::deserializePortableArtifact(
        atom_program_proto.program(), &context);
    if (!atom_program_module) {
      module->emitOpError()
          << "Failed to deserialize atom program `" << atom_program_proto.name()
          << "` from VHLO version " << atom_program_proto.version();
      signalPassFailure();
      return;
    }
    // Add the module at the end of the IFRT IR module.
    // Note: this assumes that the atom program modules are all in the top-level
    // IFRT IR module, which is what it is currently supported.
    builder.setInsertionPointToEnd(module.getBody());
    CloneModuleUsingBuilder(atom_program_module.get(), builder);
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtAtomProgramsFromVhloPass(
    const tsl::protobuf::RepeatedPtrField<IfrtIrAtomProgramProto>&
        atom_programs) {
  return std::make_unique<IfrtAtomProgramsFromVhloPass>(atom_programs);
}

}  // namespace ifrt
}  // namespace xla
