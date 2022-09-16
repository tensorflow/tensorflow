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

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace quant {
namespace {

constexpr absl::string_view kMainFunctionName = "main";

// This pass moves all ops from initializer functions to the main function. The
// main function's tf_executor::GraphOp fetches all the control outputs from the
// initializer functions.
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
           "The main function's FetchOp will output all the control outputs "
           "from the initializer functions. The initializer functions will be "
           "removed after this pass.";
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
      StringAttr::get(module_op.getContext(), kMainFunctionName);
  auto func_ops = module_op.getOps<func::FuncOp>();
  auto main_func_itr = absl::c_find_if(func_ops, [&main_func_id](auto func_op) {
    return func_op.getName() == main_func_id;
  });

  if (main_func_itr == func_ops.end()) return {};
  return *main_func_itr;
}

// Gets the GraphOp from the function op. Returns an empty op iff it doesn't
// exist.
tf_executor::GraphOp GetGraphOpFromFuncOp(func::FuncOp func_op) {
  if (func_op->getNumRegions() == 0 || func_op.getBody().empty()) {
    return {};
  }

  auto graph_op_range = func_op.front().without_terminator();
  if (llvm::hasSingleElement(graph_op_range)) {
    // The pass runs on a valid tf_executor dialect, so the op should be the
    // GraphOp.
    return cast<tf_executor::GraphOp>(graph_op_range.begin());
  }

  return {};
}

// All arguments in the initializer function should not have zero uses.
LogicalResult ValidateInitFuncArguments(func::FuncOp init_func_op) {
  for (BlockArgument arg : init_func_op.getArguments()) {
    if (!arg.use_empty()) {
      const int arg_idx = arg.getArgNumber();
      const int num_uses = absl::c_distance(arg.getUses());
      init_func_op.emitError(
          absl::StrCat("Validation failed for the initializer function: ",
                       init_func_op.getName().str(),
                       ". The initializer function's arguments should have no "
                       "usages. Instead, argument index: ",
                       arg_idx, " has number of usages: ", num_uses, "."));
      return failure();
    }
  }
  return success();
}

// Retrieves the initializer functions. The initializer functions are validated
// for whether it can be moved to the main function. Returns failure() iff
// validation fails.
FailureOr<llvm::SmallVector<func::FuncOp>> GetInitFuncOps(
    tf_saved_model::SessionInitializerOp session_init_op,
    SymbolTable symbol_table) {
  const auto initializer_symbol_refs =
      session_init_op.initializersAttr().getAsValueRange<FlatSymbolRefAttr>();

  const auto lookup_func_op =
      [symbol_table](const auto initializer_symbol_ref) {
        return symbol_table.lookup<func::FuncOp>(initializer_symbol_ref);
      };

  llvm::SmallVector<func::FuncOp> init_func_ops{};
  absl::c_transform(initializer_symbol_refs, std::back_inserter(init_func_ops),
                    lookup_func_op);

  for (auto init_func_op : init_func_ops) {
    if (failed(ValidateInitFuncArguments(init_func_op))) {
      return failure();
    }
  }

  return init_func_ops;
}

// Copies ops from `src_func_op` to `main_body` and returns the cloned FetchOp
// that corresponds to `src_func_op`'s FetchOp. If `src_func_op` is empty, it
// returns an empty op. The `main_body` after running this function contains two
// `FetchOp`s, which are the original fetch op and the cloned fetch op.
tf_executor::FetchOp CopyOpsToMainFunctionBody(func::FuncOp src_func_op,
                                               Block& main_body) {
  tf_executor::GraphOp src_graph_op = GetGraphOpFromFuncOp(src_func_op);
  if (!src_graph_op) {
    VLOG(1) << "Function " << src_func_op.getName().str()
            << " does not have a tf_executor::GraphOp. No ops are copied to "
               "the main function.";
    return {};
  }

  // Clones each op from src to main_body.
  Block& src_body = src_graph_op.GetBody();
  // TODO(b/245473863): Handle when assets are actually used in the body.
  BlockAndValueMapping mapper{};
  for (Operation& op : src_body.without_terminator()) {
    main_body.push_back(op.clone(mapper));
  }

  // Copies the fetch op into the body.
  auto cloned_fetch_op = src_body.getTerminator()->clone(mapper);
  main_body.push_back(cloned_fetch_op);

  return cast<tf_executor::FetchOp>(cloned_fetch_op);
}

// Combines the fetches from multiple fetch ops and builds a new fetch op. The
// original fetch ops will be erased.
// TODO(b/244253788): Merge the control outputs from multiple initializer
// functions to a single NoOp with control dependencies.
void MergeFetchOps(const Location loc, OpBuilder builder,
                   const ArrayRef<tf_executor::FetchOp> fetch_ops) {
  llvm::SmallVector<Value> fetches{};
  absl::c_for_each(fetch_ops, [&fetches](auto fetch_op) {
    fetches.append(fetch_op.fetches().begin(), fetch_op.fetches().end());
  });

  builder.create<tf_executor::FetchOp>(loc, fetches);

  // Erase the fetch ops once the merged FetchOp is created.
  absl::c_for_each(fetch_ops, [](auto fetch_op) { fetch_op.erase(); });
}

// Moves `src_func_op`'s ops into the main function's graph op, essentially
// merging the two `GraphOp`s. The resulting GraphOp returns all of the original
// main function's return values and the source function's return values.
void MoveOpsToMainFunction(func::FuncOp src_func_op,
                           tf_executor::GraphOp main_graph_op) {
  // Existing fetch op. Will be replaced by the updated FetchOp.
  Block& main_body = main_graph_op.GetBody();
  auto main_fetch_op = cast<tf_executor::FetchOp>(main_body.getTerminator());

  tf_executor::FetchOp src_fetch_op =
      CopyOpsToMainFunctionBody(src_func_op, main_body);
  if (!src_fetch_op) return;

  auto builder = OpBuilder::atBlockEnd(&main_body);
  // TODO(b/245448931): Prepend a suffix to the location and add tests.
  MergeFetchOps(main_graph_op.getLoc(), builder,
                /*fetch_ops=*/{main_fetch_op, src_fetch_op});
}

void ClearInitializersAttr(tf_saved_model::SessionInitializerOp session_init_op,
                           MLIRContext* ctx) {
  session_init_op.initializersAttr(
      ArrayAttr::get(ctx, llvm::ArrayRef<Attribute>{}));
}

void MergeInitializerFunctionOpsToMainPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = module_op.getContext();

  func::FuncOp main_func_op = GetMainFunction(module_op);
  if (!main_func_op) {
    module_op.emitError("Main function op not found.");
    return signalPassFailure();
  }

  tf_saved_model::SessionInitializerOp session_init_op =
      tf_saved_model::GetSessionInitializerOp(module_op);
  if (!session_init_op) return;

  const FailureOr<llvm::SmallVector<func::FuncOp>> init_func_ops =
      GetInitFuncOps(session_init_op, SymbolTable{module_op});
  if (failed(init_func_ops)) {
    module_op->emitError("Validation on initializer functions failed.");
    return signalPassFailure();
  }

  if (tf_executor::GraphOp main_graph_op = GetGraphOpFromFuncOp(main_func_op);
      main_graph_op) {
    absl::c_for_each(*init_func_ops, [main_graph_op](auto init_func_op) {
      MoveOpsToMainFunction(init_func_op, main_graph_op);
    });
  }

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
