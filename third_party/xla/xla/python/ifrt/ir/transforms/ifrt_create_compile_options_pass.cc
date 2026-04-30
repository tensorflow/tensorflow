/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/transforms/ifrt_create_compile_options_pass.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/mpmd/transforms/export/utils.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/compilation_utils.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/debug.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {

namespace {

class IfrtCreateCompileOptionsPass
    : public mlir::PassWrapper<IfrtCreateCompileOptionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfrtCreateCompileOptionsPass)

  explicit IfrtCreateCompileOptionsPass(
      CompileOptionsMap& compile_options_map,
      const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
          compile_options_overrides,
      int threshold_for_parameter_tupling)
      : compile_options_map_(compile_options_map),
        compile_options_overrides_(compile_options_overrides),
        threshold_for_parameter_tupling_(threshold_for_parameter_tupling) {}

 private:
  CompileOptionsMap& compile_options_map_;
  const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
      compile_options_overrides_;
  int threshold_for_parameter_tupling_;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTableCollection symbol_table;
    mlir::func::FuncOp func_op = GetMainFunction(module);

    auto walk_result = func_op.walk([&](CallOp call_op) {
      mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
      mlir::ModuleOp callee_module = callee->getParentOfType<mlir::ModuleOp>();
      std::string callee_name = callee_module.getSymName()->str();

      xla::CompileOptions compile_options = GetDefaultCompileOptions(
          call_op, /*enable_sharding_propagation=*/false,
          /*enable_parameter_tupling=*/
          (threshold_for_parameter_tupling_ > 0 &&
           callee.getNumArguments() > threshold_for_parameter_tupling_));

      // TODO(icgog): Move this logic to GetDefaultCompileOptions.
      if (auto reserved_hbm_bytes = callee->getAttrOfType<mlir::IntegerAttr>(
              mlir::mpmd::kReservedHbmBytes)) {
        SetReservedHbmBytes(compile_options.executable_build_options,
                            reserved_hbm_bytes.getInt());
      }

      auto mesh_name_attr =
          call_op->getAttrOfType<mlir::StringAttr>(kIfrtMeshNameAttrName);
      if (mesh_name_attr == nullptr) {
        call_op.emitError()
            << " is missing " << kIfrtMeshNameAttrName.str() << " attribute";
        return mlir::WalkResult::interrupt();
      }
      // While the users provide per-mesh compilation options, we need to
      // include callee name in the key because fragments assigned to the same
      // mesh might have different `reserved_hbm_bytes`.
      const std::string compile_options_key =
          absl::StrCat(callee_name, "_mesh_", mesh_name_attr.str());
      call_op->setAttr(
          kIfrtCompileOptionsKey,
          mlir::StringAttr::get(call_op->getContext(), compile_options_key));
      // Apply the user-provided per-mesh compile option overrides.
      if (auto option_overrides =
              compile_options_overrides_.find(mesh_name_attr.str());
          option_overrides != compile_options_overrides_.end()) {
        compile_options.env_option_overrides = option_overrides->second;
      }
      compile_options_map_.emplace(compile_options_key, compile_options);
      return mlir::WalkResult::skip();
    });

    if (walk_result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const override {
    return "ifrt-create-compile-options";
  }

  mlir::StringRef getDescription() const override {
    return "Gets the compile options for each IFRT atom program.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateIfrtCreateCompileOptionsPass(
    CompileOptionsMap& compile_options_map,
    const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
        compile_options_overrides,
    int threshold_for_parameter_tupling) {
  return std::make_unique<IfrtCreateCompileOptionsPass>(
      compile_options_map, compile_options_overrides,
      threshold_for_parameter_tupling);
}

absl::StatusOr<CompileOptionsMap> GetCompileOptions(
    mlir::ModuleOp module,
    const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
        compile_options_overrides,
    int threshold_for_parameter_tupling) {
  if (mlir::func::FuncOp main_func = GetMainFunction(module);
      !main_func->hasAttr(kIfrtFunctionAttrName)) {
    return absl::InvalidArgumentError("MLIR module is not an IFRT module.");
  }
  mlir::PassManager pm(module->getContext());
  InitPassManager(pm, "ifrt-create-compile-options");
  CompileOptionsMap compile_options_map;
  pm.addPass(CreateIfrtCreateCompileOptionsPass(
      compile_options_map, compile_options_overrides,
      threshold_for_parameter_tupling));
  StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  if (mlir::failed(pm.run(module))) {
    return diagnostic_handler.ConsumeStatus();
  }
  return compile_options_map;
}

}  // namespace ifrt
}  // namespace xla
