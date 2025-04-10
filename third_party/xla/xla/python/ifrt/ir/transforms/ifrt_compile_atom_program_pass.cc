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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/multi_threaded_atom_program_compiler.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace ifrt {

namespace {

class IfrtCompileAtomProgramPass
    : public mlir::PassWrapper<IfrtCompileAtomProgramPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit IfrtCompileAtomProgramPass(
      std::shared_ptr<AtomProgramCompiler> compiler,
      std::shared_ptr<
          absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
          compile_options_overrides,
      std::shared_ptr<AtomExecutableMap> atom_executable_map)
      : atom_program_compiler_(std::move(compiler),
                               std::move(compile_options_overrides), false),
        atom_executable_map_(std::move(atom_executable_map)) {}

  llvm::StringRef getArgument() const override {
    return "ifrt-compile-atom-program";
  }

  llvm::StringRef getDescription() const override {
    return "Compiles atom programs and lower CallOp to CallLoadedExecutableOp";
  }

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::sdy::SdyDialect>();
  }

  void runOnOperation() override;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfrtCompileAtomProgramPass);

 private:
  // Generates a LoadedExecutableOp.
  // Returns the symbol of the generated LoadedExecutableOp.
  absl::StatusOr<mlir::SymbolRefAttr> GenerateLoadedExecutableOp(
      mlir::ModuleOp module_op, absl::string_view symbol_name, CallOp call_op,
      mlir::OpBuilder& builder);

  MultiThreadedAtomProgramCompiler atom_program_compiler_;

  // Map from symbol name of LoadedExecutableOp to LoadedExecutable.
  std::shared_ptr<AtomExecutableMap> atom_executable_map_;
};

void IfrtCompileAtomProgramPass::runOnOperation() {
  mlir::SymbolTableCollection symbol_table;
  mlir::OpBuilder builder(&getContext());
  // Map from the hash of the CallOp to the compile future.
  llvm::DenseMap<CallOp, CompileFuture, IfrtCallOpInfo> call_to_compile_futures;
  mlir::ModuleOp module_op = getOperation();

  mlir::Attribute sdy_meshes_round_trip_attr =
      module_op->getAttr(kIfrtSdyMeshesRoundTripAttr);

  // Stash the errors in a MapVector, which maintains the order in which they
  // are encountered. We do not emit an error within the walk because atom
  // programs share a context and their compilations are dispatched in parallel.
  // Any error emitted here could leak into a scoped diagnostic handler used
  // while dispatching a compilation.
  llvm::MapVector<CallOp, std::string> call_op_to_error;

  // Walk and dispatch the compilations in parallel.
  module_op.walk(
      [&](CallOp call_op) -> mlir::WalkResult {
        // Do not dispatch the atom program for compilation it has already been
        // dispatched.
        if (!call_to_compile_futures.contains(call_op)) {
          mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
          auto callee_module =
              llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
          if (callee.getSymName() != kCalleeMainFuncName ||
              callee_module == nullptr) {
            // No need to clone the call op because it won't be modified if
            // any error is encountered.
            call_op_to_error.try_emplace(
                call_op,
                absl::StrCat(
                    "requires callee outlined as `", kCalleeMainFuncName.str(),
                    "` function in a ModuleOp. Actual callee name: ",
                    callee.getSymName().str(), ". Actual callee parent: ",
                    callee->getParentOp()->getName().getStringRef().str()));
            return mlir::WalkResult::advance();
          }

          if (call_op->hasAttr(kIsSdyPartitioned)) {
            // Add the meshes roundtrip attribute to the callee module if the
            // atom program was partitioned with sdy.
            if (!sdy_meshes_round_trip_attr) {
              call_op_to_error.try_emplace(
                  call_op,
                  "requires meshes roundtrip attribute to be set on the "
                  "program module if the atom program was partitioned with "
                  "sdy.");
              return mlir::WalkResult::advance();
            }
            xla::sdy::setFrontendAttribute(callee_module,
                                           xla::sdy::kMeshesRoundTripAttr,
                                           sdy_meshes_round_trip_attr);
          }

          absl::StatusOr<CompileFuture> compile_future =
              atom_program_compiler_.CompileModule(call_op, callee_module);
          if (!compile_future.ok()) {
            call_op_to_error.try_emplace(
                call_op,
                absl::StrCat(
                    "failed to dispatch compilation of atom executable: ",
                    compile_future.status().ToString()));
            return mlir::WalkResult::advance();
          }
          // Clone the CallOp because it will be modified later, but we want
          // to keep the original to be able to access the future.
          call_to_compile_futures[call_op.clone()] = *std::move(compile_future);
        }
        return mlir::WalkResult::advance();
      });

  if (call_op_to_error.empty()) {
    // Map from the hash of the CallOp to the symbol ref of the
    // LoadedExecutableOp.
    llvm::DenseMap<CallOp, mlir::SymbolRefAttr, IfrtCallOpInfo>
        call_op_to_loaded_exec_op_ref;
    // Walk, wait on compilations, and generate LoadedExecutableOps.
    module_op.walk(
        [&](CallOp call_op) -> mlir::WalkResult {
          mlir::SymbolRefAttr loaded_exec_op_ref;
          if (auto loaded_exec_op_ref_it =
                  call_op_to_loaded_exec_op_ref.find(call_op);
              loaded_exec_op_ref_it != call_op_to_loaded_exec_op_ref.end()) {
            // Reuse the symbol ref to the LoadedExecutableOp if we've already
            // created an op for the CallOp.
            loaded_exec_op_ref = loaded_exec_op_ref_it->second;
          } else {
            auto compile_result = call_to_compile_futures[call_op].Await();
            if (!compile_result.ok()) {
              call_op_to_error.try_emplace(
                  call_op,
                  absl::StrCat(
                      "failed to dispatch compilation of atom executable: ",
                      compile_result.status().ToString()));
              return mlir::WalkResult::advance();
            }
            auto callee_module = llvm::dyn_cast<mlir::ModuleOp>(
                call_op.getCalleeOp(symbol_table)->getParentOp());
            absl::StatusOr<mlir::SymbolRefAttr> symbol_ref =
                GenerateLoadedExecutableOp(callee_module, compile_result->name,
                                           call_op, builder);
            if (!symbol_ref.ok()) {
              call_op_to_error.try_emplace(
                  call_op,
                  absl::StrCat("failed to generate loaded executable op: ",
                               symbol_ref.status().ToString()));
              return mlir::WalkResult::advance();
            }
            loaded_exec_op_ref = *symbol_ref;
            // Clone the CallOp because it will be modified next, but we want to
            // keep the original to get the symbol ref for equal CallOps.
            call_op_to_loaded_exec_op_ref[call_op.clone()] = loaded_exec_op_ref;
            // Save the atom program executable to extend its lifetime.
            CHECK(atom_executable_map_
                      ->try_emplace(compile_result->name,
                                    std::move(compile_result->executable))
                      .second)
                << "Failed to insert atom program executable to map. "
                << "Executable `" << compile_result->name << "` already exists";
          }

          // Generate CallLoadedExecutableOp.
          builder.setInsertionPointAfter(call_op);
          auto new_call = builder.create<CallLoadedExecutableOp>(
              call_op.getLoc(), call_op.getResultTypes(), call_op.getInputs(),
              call_op.getControlInputs(), call_op.getArgAttrsAttr(),
              call_op.getResAttrsAttr(), loaded_exec_op_ref,
              call_op.getIoAliases(), call_op.getDonatedInputIndices());
          new_call->setDiscardableAttrs(
              call_op->getDiscardableAttrDictionary());
          call_op.replaceAllUsesWith(new_call.getResults());
          call_op.erase();
          return mlir::WalkResult::advance();
        });
    // Erase the CallOp clones that we're used as keys of the map.
    for (auto& [call_op, loaded_exec_op_ref] : call_op_to_loaded_exec_op_ref) {
      call_op.erase();
    }
  }

  if (!call_op_to_error.empty()) {
    // Wait on all compile futures to ensure that 1) the errors emitted here
    // do not leak into any scoped diagnostic handlers that might be created
    // during compilation dispatch, and 2) this->compiler_ is not accessed after
    // the pass has been destructed. We don't care if the compilations succeed
    // at this point because the pass has failed anyways.
    for (auto& [call_op, future] : call_to_compile_futures) {
      (void)future.Await();
    }
    for (auto& [call_op, error] : call_op_to_error) {
      call_op.emitError(error);
    }
    signalPassFailure();
  }
  // Erase the CallOp clones that we're used as keys of the map.
  for (auto& [call_op, future] : call_to_compile_futures) {
    call_op.erase();
  }
}

absl::StatusOr<mlir::SymbolRefAttr>
IfrtCompileAtomProgramPass::GenerateLoadedExecutableOp(
    mlir::ModuleOp module_op, absl::string_view symbol_name, CallOp call_op,
    mlir::OpBuilder& builder) {
  // Generate LoadedExecutableOp.
  llvm::SmallVector<mlir::Type, 4> input_types;
  for (const mlir::Value input : call_op.getInputs()) {
    input_types.push_back(input.getType());
  }
  llvm::SmallVector<mlir::Type, 4> output_types;
  for (const mlir::Value output : call_op.getOutputs()) {
    output_types.push_back(output.getType());
  }
  builder.setInsertionPointAfter(module_op);
  builder.create<LoadedExecutableOp>(
      module_op.getLoc(), symbol_name,
      builder.getFunctionType(input_types, output_types),
      call_op.getDevicesAttr());
  return mlir::SymbolRefAttr::get(&getContext(), symbol_name);
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtCompileAtomProgramPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map) {
  CHECK(compiler != nullptr);
  return std::make_unique<IfrtCompileAtomProgramPass>(
      std::move(compiler), std::move(compile_options_overrides),
      std::move(atom_executable_map));
}

void RegisterIfrtCompileAtomProgramPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map) {
  mlir::registerPass(
      [compiler = std::move(compiler),
       compile_options_overrides = std::move(compile_options_overrides),
       atom_executable_map =
           std::move(atom_executable_map)]() -> std::unique_ptr<mlir::Pass> {
        return CreateIfrtCompileAtomProgramPass(
            std::move(compiler), std::move(compile_options_overrides),
            std::move(atom_executable_map));
      });
}

}  // namespace ifrt
}  // namespace xla
