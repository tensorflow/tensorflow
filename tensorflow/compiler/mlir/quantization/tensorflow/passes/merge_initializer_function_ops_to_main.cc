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
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"

namespace mlir {
namespace quant {
namespace {

using ::tensorflow::kImportModelDefaultGraphFuncName;
using ::tensorflow::quantization::kInitOpNamePrefix;

// There can only be max 2 init funcs; one for variable initialization and one
// for initializing resources other than variables (e.g. tables).
constexpr int kMaxNumInitializerFunctions = 2;

// This pass moves all ops from initializer functions to the main function. A
// new `tf.NoOp` that has control dependency to the initializer function for
// non-variable resources will be created. The control output of the new
// `tf.NoOp` will be merged into the main function's `FetchOp`.
class MergeInitializerFunctionOpsToMainPass
    : public PassWrapper<MergeInitializerFunctionOpsToMainPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MergeInitializerFunctionOpsToMainPass)

  explicit MergeInitializerFunctionOpsToMainPass() {}

  StringRef getArgument() const override {
    return "quant-merge-initializer-function-ops-to-main";
  }

  StringRef getDescription() const override {
    return "Moves all ops from the initializer functions to the main function. "
           "A new `tf.NoOp` that has a control dependency to the initializer "
           "function for non-variable resources will be created. Its control "
           "output will be merged into the main function's `FetchOp`. The "
           "initializer functions will be removed after this pass.";
  }

  void runOnOperation() override;

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<TF::TensorFlowDialect, tf_executor::TensorFlowExecutorDialect,
                tf_saved_model::TensorFlowSavedModelDialect>();
  }
};

// Gets the "main" function from the module. Returns an empty op iff it doesn't
// exist.
func::FuncOp GetMainFunction(ModuleOp module_op) {
  const StringAttr main_func_id =
      StringAttr::get(module_op.getContext(), kImportModelDefaultGraphFuncName);
  auto func_ops = module_op.getOps<func::FuncOp>();
  auto main_func_itr = absl::c_find_if(func_ops, [&main_func_id](auto func_op) {
    return func_op.getName() == main_func_id;
  });

  if (main_func_itr == func_ops.end()) return {};
  return *main_func_itr;
}

// Returns true iff func_op has either no Region or the body has no Blocks.
bool IsFuncOpEmpty(func::FuncOp func_op) {
  return func_op->getNumRegions() == 0 || func_op.getBody().empty();
}

// Gets the GraphOp from the function op. Returns an empty op iff it doesn't
// exist.
tf_executor::GraphOp GetGraphOpFromFuncOp(func::FuncOp func_op) {
  if (IsFuncOpEmpty(func_op)) return {};

  auto graph_op_range = func_op.front().without_terminator();
  if (llvm::hasSingleElement(graph_op_range)) {
    // The pass runs on a valid tf_executor dialect, so the op should be the
    // GraphOp.
    return cast<tf_executor::GraphOp>(graph_op_range.begin());
  }

  return {};
}

// Gets the string representation of the type name.
std::string GetTypeName(const Type type) {
  // Gets the string representation of the type name.
  std::string type_name{};
  auto os = llvm::raw_string_ostream{type_name};
  os << type;
  return type_name;
}

// An initializer function should satisfy the follwing conditions:
// 1. The arguments should not be used.
// 2. Its GraphOp should only have control outputs.
LogicalResult ValidateInitFunc(func::FuncOp init_func_op) {
  for (BlockArgument arg : init_func_op.getArguments()) {
    if (!arg.use_empty()) {
      const int arg_idx = arg.getArgNumber();
      const int num_uses = absl::c_distance(arg.getUses());
      init_func_op.emitError(absl::StrFormat(
          "Validation failed for the initializer function: %s. "
          "The initializer function's arguments should have no "
          "usages. Instead, argument index: %d has number of usages: %d.",
          init_func_op.getName().str(), arg_idx, num_uses));
      return failure();
    }
  }

  tf_executor::GraphOp graph_op = GetGraphOpFromFuncOp(init_func_op);
  if (!graph_op) return success();  // Consider empty FuncOp valid.

  tf_executor::FetchOp fetch_op = graph_op.GetFetch();
  for (const Value fetch : fetch_op.getFetches()) {
    if (!fetch.getType().isa<tf_executor::ControlType>()) {
      fetch_op.emitError(absl::StrFormat(
          "Validation failed for the initializer function: %s. "
          "All initializer function's fetches should be "
          "tf_executor::ControlType. Got: %s.",
          init_func_op.getName().str(), GetTypeName(fetch.getType())));
      return failure();
    }
  }

  return success();
}

// Retrieves the initializer functions. The initializer functions are validated
// for whether it can be moved to the main function. Returns failure() iff
// validation fails. If successful, it will return at most
// `kMaxNumInitializerFunctions` init functions.
FailureOr<llvm::SmallVector<func::FuncOp, kMaxNumInitializerFunctions>>
GetInitFuncOps(tf_saved_model::SessionInitializerOp session_init_op,
               SymbolTable symbol_table) {
  const auto initializer_symbol_refs =
      session_init_op.getInitializersAttr()
          .getAsValueRange<FlatSymbolRefAttr>();
  if (absl::c_distance(initializer_symbol_refs) > kMaxNumInitializerFunctions) {
    session_init_op->emitError(
        absl::StrFormat("SessionInitializerOp cannot have more than %d "
                        "initializer functions. Got: %d.",
                        kMaxNumInitializerFunctions,
                        absl::c_distance(initializer_symbol_refs)));
    return failure();
  }

  const auto lookup_func_op =
      [symbol_table](const auto initializer_symbol_ref) {
        return symbol_table.lookup<func::FuncOp>(initializer_symbol_ref);
      };

  llvm::SmallVector<func::FuncOp, kMaxNumInitializerFunctions> init_func_ops{};
  absl::c_transform(initializer_symbol_refs, std::back_inserter(init_func_ops),
                    lookup_func_op);

  if (absl::c_any_of(init_func_ops, [](auto init_func_op) {
        return failed(ValidateInitFunc(init_func_op));
      })) {
    return failure();
  }

  return init_func_ops;
}

// Copies ops from `src_func_op` to `main_body` except for the FetchOps. Returns
// the fetch values in the main GraphOp corresponding to the original fetch
// values from `src_func_op`. Returns an empty vector when `src_func_op` is
// empty.
llvm::SmallVector<Value> CopyOpsToMainFunction(
    func::FuncOp src_func_op, tf_executor::GraphOp main_graph_op) {
  tf_executor::GraphOp src_graph_op = GetGraphOpFromFuncOp(src_func_op);
  if (!src_graph_op) {
    VLOG(1) << "Function " << src_func_op.getName().str()
            << " does not have a tf_executor::GraphOp. No ops are copied to "
               "the main function.";
    return {};
  }

  tf_executor::FetchOp main_fetch_op = main_graph_op.GetFetch();
  const absl::Cleanup erase_main_fetch_op = [main_fetch_op]() mutable {
    main_fetch_op.erase();
  };

  Block& main_body = main_graph_op.GetBody();

  // Clones each op from src to main_body.
  Block& src_body = src_graph_op.GetBody();
  // TODO(b/245473863): Handle when assets are actually used in the body.
  BlockAndValueMapping mapper{};
  for (Operation& op : src_body.without_terminator()) {
    main_body.push_back(op.clone(mapper));
  }

  // Relocate the main function's FetchOp at the last.
  main_body.push_back(main_fetch_op->clone(mapper));

  // Clone the source's FetchOp, but do not push to the main function's body.
  // The clone is only needed to identify the fetch operands.
  auto cloned_fetch_op =
      cast<tf_executor::FetchOp>(src_graph_op.GetFetch()->clone(mapper));
  const absl::Cleanup erase_cloned_fetch_op = [cloned_fetch_op]() mutable {
    cloned_fetch_op.erase();
  };

  const auto fetch_operands = llvm::to_vector(cloned_fetch_op.getFetches());

  return fetch_operands;
}

// An overload where it accepts multiple source FuncOps. Returns all the fetches
// from the source FuncOps.
llvm::SmallVector<Value> CopyOpsToMainFunction(
    const ArrayRef<func::FuncOp> src_func_ops,
    tf_executor::GraphOp main_graph_op) {
  llvm::SmallVector<Value> fetches{};
  absl::c_for_each(src_func_ops, [main_graph_op, &fetches](auto src_func_op) {
    const auto fetch_operands =
        CopyOpsToMainFunction(src_func_op, main_graph_op);
    fetches.append(fetch_operands);
  });

  return fetches;
}

void ClearInitializersAttr(tf_saved_model::SessionInitializerOp session_init_op,
                           MLIRContext* ctx) {
  session_init_op.setInitializersAttr(
      ArrayAttr::get(ctx, llvm::ArrayRef<Attribute>{}));
}

// Creates a new `IslandOp` that wraps a `TF::NoOp`. The `IslandOp` has control
// dependencies to the values provided.
tf_executor::IslandOp CreateNoOpWithControlDependencies(
    const Location loc, tf_executor::GraphOp main_graph_op,
    const ArrayRef<Value> control_dependencies) {
  auto builder = OpBuilder::atBlockTerminator(&main_graph_op.GetBody());

  auto wrapper_island_op = builder.create<tf_executor::IslandOp>(
      loc, /*outputs=*/TypeRange{},
      /*control=*/tf_executor::ControlType::get(builder.getContext()),
      /*controlInputs=*/control_dependencies);
  wrapper_island_op.getBody().emplaceBlock();

  // Create a NoOp inside the IslandOp.
  auto guard = OpBuilder::InsertionGuard(builder);
  builder.setInsertionPointToStart(&wrapper_island_op.GetBody());

  builder.create<TF::NoOp>(loc);
  builder.create<tf_executor::YieldOp>(loc);

  return wrapper_island_op;
}

// Adds a new fetch operand for the main function's GraphOp.
void AddFetchOperandToMain(tf_executor::GraphOp main_graph_op,
                           const Value fetch_operand) {
  tf_executor::FetchOp old_fetch = main_graph_op.GetFetch();
  const absl::Cleanup erase_old_fetch = [old_fetch]() mutable {
    old_fetch.erase();
  };

  auto fetches = llvm::to_vector(old_fetch.getFetches());
  fetches.emplace_back(fetch_operand);

  auto builder = OpBuilder::atBlockTerminator(&main_graph_op.GetBody());
  builder.create<tf_executor::FetchOp>(main_graph_op.getLoc(),
                                       std::move(fetches));
}

// Creates a new Location for the init op. This creates a loc by attaching a
// prefix `kInitOpNamePrefix` to the initializer function's name so that it is
// identifiable.
Location CreateInitOpLoc(MLIRContext* ctx, func::FuncOp init_func_ops) {
  const std::string name =
      absl::StrCat(kInitOpNamePrefix, "_", init_func_ops.getName().str());
  return NameLoc::get(StringAttr::get(ctx, name));
}

void MergeInitializerFunctionOpsToMainPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = module_op.getContext();

  func::FuncOp main_func_op = GetMainFunction(module_op);
  if (!main_func_op) {
    module_op.emitError("Main function op not found.");
    return signalPassFailure();
  }

  tf_executor::GraphOp main_graph_op = GetGraphOpFromFuncOp(main_func_op);
  if (!main_graph_op) return;

  tf_saved_model::SessionInitializerOp session_init_op =
      tf_saved_model::GetSessionInitializerOp(module_op);
  if (!session_init_op) return;

  const FailureOr<llvm::SmallVector<func::FuncOp, kMaxNumInitializerFunctions>>
      init_func_ops = GetInitFuncOps(session_init_op, SymbolTable{module_op});
  if (failed(init_func_ops)) {
    module_op->emitError("Validation on initializer functions failed.");
    return signalPassFailure();
  } else if (init_func_ops->empty()) {
    VLOG(1) << "No initializer functions found.";
    return;
  }

  const llvm::SmallVector<Value> init_op_fetches =
      CopyOpsToMainFunction(*init_func_ops, main_graph_op);
  if (init_op_fetches.empty()) {
    VLOG(1) << "No fetch values exist from initializer functions.";
    return;
  }

  // Creates a NoOp that has control dependency to the initializer function
  // for non-variables.
  const Location loc = CreateInitOpLoc(ctx, init_func_ops->back());
  tf_executor::IslandOp noop_wrapper_island_op =
      CreateNoOpWithControlDependencies(
          loc, main_graph_op,
          /*control_dependencies=*/ArrayRef<Value>{init_op_fetches.back()});

  AddFetchOperandToMain(main_graph_op,
                        /*fetch_operand=*/noop_wrapper_island_op.getControl());

  // Erase the initializer function once all ops are moved to the main function.
  absl::c_for_each(*init_func_ops,
                   [](auto init_func_op) { init_func_op.erase(); });

  ClearInitializersAttr(session_init_op, ctx);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateMergeInitializerFunctionOpsToMainPass() {
  return std::make_unique<MergeInitializerFunctionOpsToMainPass>();
}

// Registers MergeInitializerFunctionOpsToMainPass.
static PassRegistration<MergeInitializerFunctionOpsToMainPass> pass([] {
  return CreateMergeInitializerFunctionOpsToMainPass();
});

}  // namespace quant
}  // namespace mlir
