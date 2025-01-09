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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/service/hlo.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

class IfrtVerifyBoundExternalLoadedExecutablePass
    : public mlir::PassWrapper<IfrtVerifyBoundExternalLoadedExecutablePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit IfrtVerifyBoundExternalLoadedExecutablePass(
      std::shared_ptr<AtomExecutableMap> bound_executable_map)
      : bound_executable_map_(std::move(bound_executable_map)) {}

  llvm::StringRef getArgument() const override {
    return "ifrt-verify-bound-external-loaded-executable";
  }

  llvm::StringRef getDescription() const override {
    return "Verifies that the bound external LoadedExecutables have number of "
           "inputs/outputs, shape, and sharding as the corresponding externally"
           " bound LoadedExecutable";
  }

  void runOnOperation() override;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      IfrtVerifyBoundExternalLoadedExecutablePass);

 private:
  absl::Status VerifyShardingsEqual(
      llvm::ArrayRef<mlir::Type> types,
      const std::vector<xla::OpSharding>& shardings,
      absl::string_view sharding_type);

  // Map from symbol name of LoadedExecutableOp to externally bound
  // LoadedExecutable.
  std::shared_ptr<AtomExecutableMap> bound_executable_map_;
};

absl::Status IfrtVerifyBoundExternalLoadedExecutablePass::VerifyShardingsEqual(
    llvm::ArrayRef<mlir::Type> types,
    const std::vector<xla::OpSharding>& shardings,
    absl::string_view sharding_type) {
  for (const auto& it : llvm::enumerate(llvm::zip(types, shardings))) {
    const auto& [param_type, sharding] = it.value();
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        xla::HloSharding::FromProto(sharding));
    auto array_type = llvm::dyn_cast<IfrtArrayType>(param_type);
    CHECK(array_type);
    auto array_sharding =
        llvm::dyn_cast<IfrtShardingParamAttr>(array_type.getShardingAttr());
    CHECK(array_sharding);
    TF_ASSIGN_OR_RETURN(
        const xla::HloSharding hlo_type_sharding,
        xla::ifrt::support::ToHloSharding(array_sharding.getSharding()));
    if (hlo_sharding != hlo_type_sharding) {
      return absl::InvalidArgumentError(absl::StrCat(
          "expects an executable with ", sharding_type, " #", it.index(),
          " sharding ", hlo_sharding.ToString(/*include_metadata=*/false),
          ", but was bound to an executable with sharding ",
          hlo_type_sharding.ToString(/*include_metadata=*/false)));
    }
  }
  return absl::OkStatus();
}

void IfrtVerifyBoundExternalLoadedExecutablePass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  // Walk and dispatch the compilations in parallel.
  auto result = module_op.walk([&](LoadedExecutableOp loaded_exec_op)
                                   -> mlir::WalkResult {
    const auto exec_it =
        bound_executable_map_->find(loaded_exec_op.getSymName());
    if (exec_it != bound_executable_map_->end()) {
      if (loaded_exec_op.getDevices().size() !=
          exec_it->second->num_devices()) {
        return loaded_exec_op.emitOpError()
               << "expects an executable with "
               << loaded_exec_op.getDevices().size()
               << " devices, but was bound to an executable with "
               << exec_it->second->num_devices() << " devices";
      }

      auto func_type = loaded_exec_op.getFunctionType();
      if (!exec_it->second->GetParameterShardings().has_value()) {
        return loaded_exec_op.emitOpError()
               << "cannot be bound to an executable without parameter "
                  "shardings";
      }
      if (!exec_it->second->GetOutputShardings().has_value()) {
        return loaded_exec_op.emitOpError()
               << "cannot be bound to an executable without output shardings";
      }
      if (func_type.getNumInputs() !=
          exec_it->second->GetParameterShardings()->size()) {
        return loaded_exec_op.emitOpError()
               << "expects an executable with " << func_type.getNumInputs()
               << " inputs, but was bound to an executable with "
               << exec_it->second->GetParameterShardings()->size() << " inputs";
      }
      if (func_type.getNumResults() !=
          exec_it->second->GetOutputShardings()->size()) {
        return loaded_exec_op.emitOpError()
               << "expects an executable with " << func_type.getNumResults()
               << " results, but was bound to an executable with "
               << exec_it->second->GetOutputShardings()->size() << " results";
      }
      // Verify that the input and output shardings of the LoadedExecutableOp
      // are the same as the shardings of the bound executable.
      if (!exec_it->second->GetParameterShardings().has_value()) {
        return loaded_exec_op.emitOpError()
               << "cannot be bound to an executable without parameter "
                  "shardings";
      }
      if (!exec_it->second->GetOutputShardings().has_value()) {
        return loaded_exec_op.emitOpError()
               << "cannot be bound to an executable without output "
                  "shardings";
      }
      auto sharding_equal_status = VerifyShardingsEqual(
          func_type.getInputs(), *exec_it->second->GetParameterShardings(),
          "input");
      if (!sharding_equal_status.ok()) {
        return loaded_exec_op.emitOpError() << sharding_equal_status.message();
      }
      sharding_equal_status = VerifyShardingsEqual(
          func_type.getResults(), *exec_it->second->GetOutputShardings(),
          "output");
      if (!sharding_equal_status.ok()) {
        return loaded_exec_op.emitOpError() << sharding_equal_status.message();
      }
    }
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyBoundExternalLoadedExecutablePass(
    std::shared_ptr<AtomExecutableMap> bound_executable_map) {
  return std::make_unique<IfrtVerifyBoundExternalLoadedExecutablePass>(
      std::move(bound_executable_map));
}

void RegisterIfrtVerifyBoundExternalLoadedExecutablePass(
    std::shared_ptr<AtomExecutableMap> bound_executable_map) {
  mlir::registerPass(
      [bound_executable_map =
           std::move(bound_executable_map)]() -> std::unique_ptr<mlir::Pass> {
        return CreateIfrtVerifyBoundExternalLoadedExecutablePass(
            std::move(bound_executable_map));
      });
}

}  // namespace ifrt
}  // namespace xla
