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
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
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
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/ir/transforms/multi_threaded_atom_program_compiler.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/service/hlo.pb.h"

namespace xla {
namespace ifrt {

namespace {

class IfrtCompileAndPropagateShardingsPass
    : public mlir::PassWrapper<IfrtCompileAndPropagateShardingsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit IfrtCompileAndPropagateShardingsPass(
      std::shared_ptr<AtomProgramCompiler> compiler,
      std::shared_ptr<
          absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
          compile_options_overrides,
      std::shared_ptr<AtomExecutableMap> atom_executable_map)
      : atom_program_compiler_(std::move(compiler),
                               std::move(compile_options_overrides), true),
        atom_executable_map_(std::move(atom_executable_map)) {}

  llvm::StringRef getArgument() const override {
    return "ifrt-compile-and-propagate-shardings";
  }

  llvm::StringRef getDescription() const override {
    return "Compiles atom programs, propagates shardings to infer unspecified "
           " shardings, and lowers CallOp to CallLoadedExecutableOp";
  }

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
  }

  void runOnOperation() override;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      IfrtCompileAndPropagateShardingsPass);

 private:
  mlir::LogicalResult WalkCallOp(
      CallOp call_op, mlir::SymbolTableCollection& symbol_table,
      mlir::OpBuilder& builder,
      llvm::DenseMap<mlir::ModuleOp,
                     std::pair<AtomProgramCompileResult, mlir::SymbolRefAttr>>&
          module_to_compiled);

  // Propagates sharding to/from a ReshardOp.
  //
  // Cases:
  // 1) All shardings are specified => no=op.
  // 2) Output sharding is not specified => error.
  // 3) Input sharding not specified, but output sharding is specified => the
  // output sharding is propagated to the input. This case is only possible
  // if the input is an argument of the main func, and thus the input sharding
  // is propagated to the other ops using this argument. For example, in the
  // following case:
  // ```
  // func.func @main(%arg0: !array_no_sharding) -> () {
  //   %0, %ctrl_0 = ifrt.Reshard(%arg0)
  //      : (!array_no_sharding) -> !array_with_sharding
  //   %1, %ctrl_1 = ifrt.Call @program::@main(%arg0) on devices [0, 1]
  //       : (!array_no_sharding) -> !array_no_sharding
  // }
  // ```
  // The Reshard's output sharding will be propagated to the `@program`
  // input. Thus, the order in which the ops appear in the IR, has significant
  // effect on the shardings inferred.
  mlir::LogicalResult WalkReshardOp(ReshardOp reshard_op);

  // See the documentation of `WalkReshardOp`.
  mlir::LogicalResult WalkCopyArraysOp(CopyArraysOp copy_arrays_op);

  mlir::LogicalResult WalkCopyArraysOrReshardOp(mlir::Operation* op,
                                                mlir::OperandRange inputs,
                                                mlir::ResultRange outputs);

  // Replaces all the unspecified input shardings attributes in the atom
  // program's main func.
  mlir::LogicalResult LowerInputShardingToXla(CallOp call_op,
                                              mlir::func::FuncOp callee,
                                              mlir::OpBuilder& builder);

  // Returns a vector of `ShardingParam` for inputs.
  //
  // If an input sharding is unspecified in the IFRT IR op, then the sharding
  // is fetched from the compiled executable. Otherwise, the sharding present in
  // the IFRT IR op is returned.
  mlir::FailureOr<llvm::SmallVector<ShardingParam>> GetInputShardingParams(
      CallOp call_op, const AtomProgramCompileResult& compile_result);

  // Returns a vector of `ShardingParam` for outputs.
  //
  // If an output sharding is unspecified in the IFRT IR op, then the sharding
  // is fetched from the compiled executable. Otherwise, the sharding present in
  // the IFRT IR op is returned.
  mlir::FailureOr<llvm::SmallVector<ShardingParam>> GetOutputShardingParams(
      CallOp call_op, const AtomProgramCompileResult& compile_result);

  // The method does the following:
  // 1) Populates the unspecified sharding attributes in the atom program's main
  // func.
  // 2) Replaces the sharding in op's input types with unspecified sharding.
  // 3) Replaces op's outputs with unspecified shardings.
  // 4) Replaces all the usage of replaced outputs.
  mlir::FailureOr<CallOp> PropagateShardings(
      CallOp call_op, mlir::func::FuncOp callee,
      llvm::ArrayRef<ShardingParam> input_shardings,
      llvm::ArrayRef<ShardingParam> output_shardings, mlir::OpBuilder& builder);

  // Generates a LoadedExecutableOp.
  // Returns the symbol of the generated LoadedExecutableOp.
  mlir::FailureOr<mlir::SymbolRefAttr> GenerateLoadedExecutableOp(
      mlir::ModuleOp module_op, absl::string_view symbol_name, CallOp call_op,
      mlir::OpBuilder& builder);

  // Replaces the CallOp with a CallLoadedExecutableOp.
  void ReplaceCallOpWithCallLoadedOp(CallOp call_op,
                                     mlir::SymbolRefAttr loaded_exec_op_callee,
                                     mlir::OpBuilder& builder);

  MultiThreadedAtomProgramCompiler atom_program_compiler_;

  // Map from symbol name of LoadedExecutableOp to LoadedExecutable.
  std::shared_ptr<AtomExecutableMap> atom_executable_map_;
};

mlir::LogicalResult IfrtCompileAndPropagateShardingsPass::WalkCallOp(
    CallOp call_op, mlir::SymbolTableCollection& symbol_table,
    mlir::OpBuilder& builder,
    llvm::DenseMap<mlir::ModuleOp,
                   std::pair<AtomProgramCompileResult, mlir::SymbolRefAttr>>&
        module_to_compiled) {
  mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
  auto callee_module = llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
  if (callee.getSymName() != kCalleeMainFuncName || callee_module == nullptr) {
    return call_op.emitOpError()
           << "requires callee outlined as `" << kCalleeMainFuncName
           << "` function in a ModuleOp. Actual callee name: "
           << callee.getSymName()
           << ". Actual callee parent: " << callee->getParentOp()->getName();
  }

  if (auto compiled_it = module_to_compiled.find(callee_module);
      compiled_it == module_to_compiled.end()) {
    // Only dispatch for compilation if the module has not been dispatched yet.
    if (mlir::failed(LowerInputShardingToXla(call_op, callee, builder)))
      return mlir::failure();

    absl::StatusOr<CompileFuture> compile_future =
        atom_program_compiler_.CompileModule(call_op, callee_module);
    if (!compile_future.ok()) {
      return call_op.emitOpError()
             << "failed to dispatch compilation for atom executable: "
             << compile_future.status().message();
    }
    auto compile_result = compile_future->Await();
    if (!compile_result.ok()) {
      return call_op.emitOpError() << "failed to compile to atom executable: "
                                   << compile_result.status().message();
    }

    // Get the input and output shardings from the compiled executable. Only
    // the unspecified shardings are fetched from the executable, all the other
    // shardings remain the same.
    auto input_shardings = GetInputShardingParams(call_op, *compile_result);
    if (mlir::failed(input_shardings)) return mlir::failure();
    auto output_shardings = GetOutputShardingParams(call_op, *compile_result);
    if (mlir::failed(output_shardings)) return mlir::failure();
    // Change the CallOp signature and propagate shardings.
    auto new_call_op = PropagateShardings(call_op, callee, *input_shardings,
                                          *output_shardings, builder);
    if (mlir::failed(new_call_op)) return mlir::failure();

    // Create a LoadedExecutableOp for the atom program.
    auto symbol_ref = GenerateLoadedExecutableOp(
        callee_module, compile_result->name, *new_call_op, builder);
    if (mlir::failed(symbol_ref)) return mlir::failure();

    // Save the atom program executable to extend its lifetime.
    CHECK(atom_executable_map_
              ->try_emplace(compile_result->name, compile_result->executable)
              .second);
    CHECK(module_to_compiled
              .try_emplace(callee_module,
                           std::make_pair(*compile_result, *symbol_ref))
              .second);
    ReplaceCallOpWithCallLoadedOp(*new_call_op, *symbol_ref, builder);
  } else {
    // The module has been compiled already. Get the unspecified input and
    // output shardings from the compiled executable to set the shardings in the
    // CallOp and to propagate them.
    auto input_shardings =
        GetInputShardingParams(call_op, compiled_it->second.first);
    if (mlir::failed(input_shardings)) return mlir::failure();
    auto output_shardings =
        GetOutputShardingParams(call_op, compiled_it->second.first);
    if (mlir::failed(output_shardings)) return mlir::failure();
    auto new_call_op = PropagateShardings(call_op, callee, *input_shardings,
                                          *output_shardings, builder);
    if (mlir::failed(new_call_op)) return mlir::failure();
    ReplaceCallOpWithCallLoadedOp(*new_call_op, compiled_it->second.second,
                                  builder);
  }
  return mlir::success();
}

mlir::LogicalResult IfrtCompileAndPropagateShardingsPass::WalkReshardOp(
    ReshardOp reshard_op) {
  return WalkCopyArraysOrReshardOp(reshard_op, reshard_op.getInputs(),
                                   reshard_op.getOutputs());
}

mlir::LogicalResult IfrtCompileAndPropagateShardingsPass::WalkCopyArraysOp(
    CopyArraysOp copy_arrays_op) {
  return WalkCopyArraysOrReshardOp(copy_arrays_op, copy_arrays_op.getInputs(),
                                   copy_arrays_op.getOutputs());
}

mlir::LogicalResult
IfrtCompileAndPropagateShardingsPass::WalkCopyArraysOrReshardOp(
    mlir::Operation* op, mlir::OperandRange inputs, mlir::ResultRange outputs) {
  for (auto [idx, pair] : llvm::enumerate(llvm::zip(inputs, outputs))) {
    auto in_array_type = mlir::cast<IfrtArrayType>(std::get<0>(pair).getType());
    if (in_array_type == nullptr) {
      op->emitOpError() << "requires all inputs to be `IfrtArrayType`. Input #"
                        << idx << ": " << std::get<0>(pair).getType();
      return mlir::failure();
    }
    auto out_array_type =
        mlir::cast<IfrtArrayType>(std::get<1>(pair).getType());
    if (out_array_type == nullptr) {
      op->emitOpError()
          << "requires all outputs to be `IfrtArrayType`. Output #" << idx
          << ": " << std::get<1>(pair).getType();
      return mlir::failure();
    }
    if (mlir::isa<IfrtUnspecifiedShardingAttr>(
            in_array_type.getShardingAttr())) {
      if (llvm::isa<IfrtUnspecifiedShardingAttr>(
              out_array_type.getShardingAttr())) {
        return op->emitOpError()
               << "requires output #" << idx << " to have sharding specified.";
      } else {
        std::get<0>(pair).setType(out_array_type);
      }
    }
  }
  return mlir::success();
}

void IfrtCompileAndPropagateShardingsPass::runOnOperation() {
  mlir::SymbolTableCollection symbol_table;
  mlir::OpBuilder builder(&getContext());
  mlir::ModuleOp module_op = getOperation();
  llvm::DenseMap<mlir::ModuleOp,
                 std::pair<AtomProgramCompileResult, mlir::SymbolRefAttr>>
      module_to_compiled;
  auto compile_result =
      module_op.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (mlir::isa<CallOp>(op)) {
          if (mlir::failed(WalkCallOp(mlir::cast<CallOp>(op), symbol_table,
                                      builder, module_to_compiled)))
            return mlir::WalkResult::interrupt();
        } else if (mlir::isa<ReshardOp>(op)) {
          if (mlir::failed(WalkReshardOp(mlir::cast<ReshardOp>(op)))) {
            return mlir::WalkResult::interrupt();
          }
        } else if (mlir::isa<CopyArraysOp>(op)) {
          if (mlir::failed(WalkCopyArraysOp(mlir::cast<CopyArraysOp>(op)))) {
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      });

  if (compile_result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // Update the main function's result types.
  mlir::func::FuncOp main_func = GetMainFunction(module_op);
  UpdateFunctionType(main_func);
}

mlir::LogicalResult
IfrtCompileAndPropagateShardingsPass::LowerInputShardingToXla(
    CallOp call_op, mlir::func::FuncOp callee, mlir::OpBuilder& builder) {
  CHECK_EQ(call_op.getInputs().size(), callee.getNumArguments());
  for (int idx = 0; idx < callee.getNumArguments(); ++idx) {
    const auto hlo_sharding_attr =
        callee.getArgAttrOfType<mlir::StringAttr>(idx, kHloShardingAttrName);
    if (hlo_sharding_attr == nullptr) {
      // The input sharding is not set, see if it has been inferred and can
      // be lowered from IFRT Sharding to HloSharding.
      auto array_type = mlir::dyn_cast_or_null<IfrtArrayType>(
          call_op.getInputs()[idx].getType());
      auto sharding_param_attr = mlir::dyn_cast_or_null<IfrtShardingParamAttr>(
          array_type.getShardingAttr());
      if (sharding_param_attr != nullptr) {
        // Set the newly inferred sharding on the callee's argument.
        auto hlo_sharding = xla::ifrt::support::ToHloSharding(
            sharding_param_attr.getSharding());
        callee.setArgAttr(
            idx, kHloShardingAttrName,
            builder.getStringAttr(hlo_sharding.value().ToString()));
      }
    }
  }
  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<ShardingParam>>
IfrtCompileAndPropagateShardingsPass::GetInputShardingParams(
    CallOp call_op, const AtomProgramCompileResult& compile_result) {
  std::optional<std::vector<xla::OpSharding>> in_shardings = std::nullopt;
  llvm::SmallVector<ShardingParam> in_sharding_params;
  in_sharding_params.reserve(call_op.getInputs().size());
  for (const auto& [idx, input] : llvm::enumerate(call_op.getInputs())) {
    const auto in_array_type =
        mlir::dyn_cast_or_null<IfrtArrayType>(input.getType());
    // Get sharding from the executable if it is unspecified in the op.
    if (llvm::isa<IfrtUnspecifiedShardingAttr>(
            in_array_type.getShardingAttr())) {
      if (!in_shardings.has_value()) {
        in_shardings = compile_result.executable->GetParameterShardings();
        if (!in_shardings.has_value()) {
          return call_op.emitError()
                 << "executable does not have input shardings";
        }
        if (in_shardings->size() != call_op.getOutputs().size()) {
          return call_op.emitError()
                 << "mismatch between number of input shardings of executable "
                    "and op's return: "
                 << in_shardings->size() << " vs. "
                 << call_op.getOutputs().size();
        }
      }
      auto hlo_sharding = xla::HloSharding::FromProto(in_shardings->at(idx));
      if (!hlo_sharding.ok()) {
        return call_op.emitError()
               << "failed to convert sharding `OpSharding` to `HloSharding`: "
               << hlo_sharding.status().message();
      }
      auto sharding_param = xla::ifrt::support::ToShardingParam(
          hlo_sharding.value(), in_array_type.getShape().getRank(),
          in_array_type.getDevices().size());
      if (!sharding_param.ok()) {
        return call_op.emitError() << "failed to convert sharding "
                                      "`HloSharding` to `ShardingParam`: "
                                   << sharding_param.status().message();
      }
      in_sharding_params.push_back(std::move(sharding_param.value()));
    } else {
      auto sharding_param_attr = mlir::dyn_cast_or_null<IfrtShardingParamAttr>(
          in_array_type.getShardingAttr());
      if (sharding_param_attr == nullptr) {
        return call_op.emitError() << "input #" << idx
                                   << " has sharding attribute that is not of "
                                      "type `IfrtShardingParamAttr`.";
      }
      in_sharding_params.push_back(sharding_param_attr.getSharding());
    }
  }
  return in_sharding_params;
}

mlir::FailureOr<llvm::SmallVector<ShardingParam>>
IfrtCompileAndPropagateShardingsPass::GetOutputShardingParams(
    CallOp call_op, const AtomProgramCompileResult& compile_result) {
  std::optional<std::vector<xla::OpSharding>> out_shardings = std::nullopt;
  llvm::SmallVector<ShardingParam> out_sharding_params;
  out_sharding_params.reserve(call_op.getOutputs().size());
  for (const auto& [idx, output] : llvm::enumerate(call_op.getOutputs())) {
    const auto out_array_type =
        mlir::dyn_cast_or_null<IfrtArrayType>(output.getType());
    // Get sharding from the executable if it is unspecified in the op.
    if (llvm::isa<IfrtUnspecifiedShardingAttr>(
            out_array_type.getShardingAttr())) {
      if (!out_shardings.has_value()) {
        out_shardings = compile_result.executable->GetOutputShardings();
        if (!out_shardings.has_value()) {
          return call_op.emitError()
                 << "executable does not have output shardings";
        }
        if (out_shardings->size() != call_op.getOutputs().size()) {
          return call_op.emitError()
                 << "mismatch between number of output shardings of executable "
                    "and op's return: "
                 << out_shardings->size() << " vs. "
                 << call_op.getOutputs().size();
        }
      }
      auto hlo_sharding = xla::HloSharding::FromProto(out_shardings->at(idx));
      if (!hlo_sharding.ok()) {
        return call_op.emitError()
               << "failed to convert sharding `OpSharding` to `HloSharding`: "
               << hlo_sharding.status().message();
      }
      auto sharding_param = xla::ifrt::support::ToShardingParam(
          hlo_sharding.value(), out_array_type.getShape().getRank(),
          out_array_type.getDevices().size());
      if (!sharding_param.ok()) {
        return call_op.emitError() << "failed to convert sharding "
                                      "`HloSharding` to `ShardingParam`: "
                                   << sharding_param.status().message();
      }
      out_sharding_params.push_back(std::move(sharding_param.value()));
    } else {
      auto sharding_param_attr = mlir::dyn_cast_or_null<IfrtShardingParamAttr>(
          out_array_type.getShardingAttr());
      if (sharding_param_attr == nullptr) {
        return call_op.emitError() << "output #" << idx
                                   << " has sharding attribute that is not of "
                                      "type `IfrtShardingParamAttr`.";
      }
      out_sharding_params.push_back(sharding_param_attr.getSharding());
    }
  }
  return out_sharding_params;
}

mlir::FailureOr<CallOp>
IfrtCompileAndPropagateShardingsPass::PropagateShardings(
    CallOp call_op, mlir::func::FuncOp callee,
    llvm::ArrayRef<ShardingParam> input_shardings,
    llvm::ArrayRef<ShardingParam> output_shardings, mlir::OpBuilder& builder) {
  CHECK_EQ(call_op.getOutputs().size(), callee.getNumResults());
  CHECK_EQ(input_shardings.size(), callee.getNumArguments());
  CHECK_EQ(output_shardings.size(), callee.getNumResults());

  // Compute arg types. An unspecified sharding is replaced with its
  // corresponding input sharding.
  for (const auto& [idx, input] : llvm::enumerate(call_op.getInputs())) {
    const auto array_type =
        mlir::dyn_cast_or_null<IfrtArrayType>(input.getType());
    const auto unspecified_sharding_attr =
        callee.getArgAttrOfType<mlir::StringAttr>(idx, kHloShardingAttrName);
    if (unspecified_sharding_attr == nullptr) {
      auto hlo_sharding =
          xla::ifrt::support::ToHloSharding(input_shardings[idx]);
      if (!hlo_sharding.ok()) {
        return call_op.emitOpError()
               << "can't lower sharding of input #" << idx
               << ". Sharding: " << input_shardings[idx].DebugString() << ". "
               << hlo_sharding.status().message();
      }
      callee.setArgAttr(idx, kHloShardingAttrName,
                        builder.getStringAttr(hlo_sharding.value().ToString()));
      auto new_array_type = builder.getType<IfrtArrayType>(
          array_type.getShape(),
          IfrtShardingParamAttr::get(builder.getContext(),
                                     input_shardings[idx]),
          array_type.getDevicesAttr(), array_type.getMemoryKindAttr(),
          array_type.getLayoutAttr());
      input.setType(new_array_type);
    }
  }

  // Compute result types. An unspecified sharding is replaced with its
  // corresponding output sharding.
  llvm::SmallVector<mlir::Type, 4> new_call_op_result_types;
  new_call_op_result_types.reserve(call_op.getOutputs().size());
  // The op must be replaced if at least an output sharding needs to be changed.
  bool replace_call_op = false;
  for (const auto& [idx, output] : llvm::enumerate(call_op.getOutputs())) {
    if (callee.getResultAttrOfType<mlir::StringAttr>(
            idx, kHloShardingAttrName) == nullptr) {
      auto hlo_sharding =
          xla::ifrt::support::ToHloSharding(output_shardings[idx]);
      if (!hlo_sharding.ok()) {
        return call_op.emitOpError()
               << "can't lower sharding of output #" << idx
               << ". Sharding: " << output_shardings[idx].DebugString() << ". "
               << hlo_sharding.status().message();
      }
      callee.setResultAttr(
          idx, kHloShardingAttrName,
          builder.getStringAttr(hlo_sharding.value().ToString()));
    }

    const auto array_type =
        mlir::dyn_cast_or_null<IfrtArrayType>(output.getType());
    // If the CallOp has an output with an unspecified sharding, then the
    // CallOp must be replaced with a new CallOp that has the propagated
    // shardings.
    if (mlir::isa<IfrtUnspecifiedShardingAttr>(array_type.getShardingAttr())) {
      replace_call_op = true;
      auto new_array_type = builder.getType<IfrtArrayType>(
          array_type.getShape(),
          IfrtShardingParamAttr::get(builder.getContext(),
                                     output_shardings[idx]),
          array_type.getDevicesAttr(), array_type.getMemoryKindAttr(),
          array_type.getLayoutAttr());
      new_call_op_result_types.push_back(new_array_type);
    } else {
      new_call_op_result_types.push_back(output.getType());
    }
  }

  if (replace_call_op) {
    builder.setInsertionPointAfter(call_op);
    auto new_call_op = builder.create<CallOp>(
        call_op.getLoc(), /*outputs=*/new_call_op_result_types,
        /*control_output=*/builder.getType<IfrtControlType>(),
        /*inputs=*/call_op.getInputs(),
        /*control_inputs=*/call_op.getControlInputs(),
        /*callee=*/call_op.getCallee(),
        /*devices=*/call_op.getDevices(),
        /*io_aliases=*/call_op.getIoAliases(),
        /*donated_input_indices=*/call_op.getDonatedInputIndices());
    new_call_op->setDiscardableAttrs(call_op->getDiscardableAttrDictionary());
    for (auto [i, result] : llvm::enumerate(call_op.getOutputs())) {
      result.replaceAllUsesWith(new_call_op.getOutputs()[i]);
    }
    call_op.getControlOutput().replaceAllUsesWith(
        new_call_op.getControlOutput());
    call_op.erase();
    return new_call_op;
  } else {
    return call_op;
  }
}

mlir::FailureOr<mlir::SymbolRefAttr>
IfrtCompileAndPropagateShardingsPass::GenerateLoadedExecutableOp(
    mlir::ModuleOp module_op, absl::string_view symbol_name, CallOp call_op,
    mlir::OpBuilder& builder) {
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

void IfrtCompileAndPropagateShardingsPass::ReplaceCallOpWithCallLoadedOp(
    CallOp call_op, mlir::SymbolRefAttr loaded_exec_op_callee,
    mlir::OpBuilder& builder) {
  builder.setInsertionPointAfter(call_op);
  auto call_loaded_op = builder.create<CallLoadedExecutableOp>(
      call_op.getLoc(), call_op.getResultTypes(), call_op.getInputs(),
      call_op.getControlInputs(), loaded_exec_op_callee, call_op.getIoAliases(),
      call_op.getDonatedInputIndices());
  call_op.replaceAllUsesWith(call_loaded_op.getResults());
  call_op.erase();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtCompileAndPropagateShardingsPass(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map) {
  CHECK(compiler != nullptr);
  return std::make_unique<IfrtCompileAndPropagateShardingsPass>(
      std::move(compiler), std::move(compile_options_overrides),
      std::move(atom_executable_map));
}

void RegisterIfrtCompileAndPropagateShardingsPass(
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
        return CreateIfrtCompileAndPropagateShardingsPass(
            std::move(compiler), std::move(compile_options_overrides),
            std::move(atom_executable_map));
      });
}

}  // namespace ifrt
}  // namespace xla
