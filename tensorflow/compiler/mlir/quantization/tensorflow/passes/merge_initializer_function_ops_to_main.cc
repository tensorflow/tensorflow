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
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
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

using ::mlir::tf_executor::FetchOp;
using ::mlir::tf_executor::GraphOp;
using ::mlir::tf_executor::IslandOp;
using ::mlir::tf_saved_model::GetSessionInitializerOp;
using ::mlir::tf_saved_model::kTfSavedModelInitializerInitType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerTypeAttr;
using ::mlir::tf_saved_model::SessionInitializerOp;
using ::tensorflow::kImportModelDefaultGraphFuncName;
using ::tensorflow::quantization::kInitOpNamePrefix;

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

  explicit MergeInitializerFunctionOpsToMainPass() = default;

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
  const auto main_func_id =
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
GraphOp GetGraphOpFromFuncOp(func::FuncOp func_op) {
  if (IsFuncOpEmpty(func_op)) return {};

  auto graph_op_range = func_op.front().without_terminator();
  if (llvm::hasSingleElement(graph_op_range)) {
    // The pass runs on a valid tf_executor dialect, so the op should be the
    // GraphOp.
    return cast<GraphOp>(graph_op_range.begin());
  }

  return {};
}

// Gets the string representation of the type name.
std::string GetTypeName(const Type type) {
  std::string type_name{};
  auto os = llvm::raw_string_ostream{type_name};
  os << type;
  return type_name;
}

// Retrieves the value of `tf_saved_model.initializer_type` attribute from the
// initializer function. Returns "unknown_initializer_type" iff the attribute is
// not set.
std::string GetInitializerType(func::FuncOp init_func_op) {
  const auto initializer_type_attr =
      init_func_op->getAttrOfType<StringAttr>(kTfSavedModelInitializerTypeAttr);

  if (!initializer_type_attr) {
    init_func_op->emitWarning()
        << "Initializer func op does not have tf_saved_model.initializer_type "
           "attribute. Func op: "
        << init_func_op.getSymName();
    return "unknown_initializer_type";
  }

  return initializer_type_attr.str();
}

// An initializer function should satisfy the follwing conditions:
// * Its GraphOp should only have control outputs.
LogicalResult ValidateInitFunc(func::FuncOp init_func_op) {
  GraphOp graph_op = GetGraphOpFromFuncOp(init_func_op);
  if (!graph_op) return success();  // Consider empty FuncOp valid.

  FetchOp fetch_op = graph_op.GetFetch();
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

// Returns initializer_type -> init_func_op mapping from the session_init_op's
// initializers. The initializer functions are validated for whether it can be
// moved to the main function. Returns failure() iff validation fails.
FailureOr<absl::flat_hash_map<std::string, func::FuncOp>> GetInitFuncOps(
    SessionInitializerOp session_init_op, SymbolTable symbol_table) {
  const auto initializer_symbol_refs =
      session_init_op.getInitializersAttr()
          .getAsValueRange<FlatSymbolRefAttr>();

  absl::flat_hash_map<std::string, func::FuncOp> init_func_ops;

  for (auto initializer_symbol_ref : initializer_symbol_refs) {
    auto init_func_op =
        symbol_table.lookup<func::FuncOp>(initializer_symbol_ref);

    if (failed(ValidateInitFunc(init_func_op))) {
      return failure();
    }

    const std::string initializer_type = GetInitializerType(init_func_op);
    init_func_ops[initializer_type] = init_func_op;
  }

  return init_func_ops;
}

// If `main_func_op` has the `tf.entry_function` attribute, adds a new input
// name to the `inputs` field of the attribute. Otherwise, no attribute is
// modified.
void MaybeAddEntryFunctionInput(const StringRef input_name,
                                func::FuncOp main_func_op) {
  auto entry_func_attr =
      main_func_op->getAttrOfType<DictionaryAttr>("tf.entry_function");
  if (!entry_func_attr) return;

  auto entry_func_attrs = SmallVector<NamedAttribute>(entry_func_attr.begin(),
                                                      entry_func_attr.end());

  MLIRContext* ctx = main_func_op.getContext();
  for (auto& named_attr : entry_func_attrs) {
    if (named_attr.getName() != "inputs") continue;

    // Splits the "inputs" field to retrieve individual input names. Ignores
    // empty strings.
    SmallVector<StringRef> inputs_attrs{};
    cast<StringAttr>(named_attr.getValue())
        .strref()
        .split(inputs_attrs, /*Separator=*/',', /*MaxSplit=*/-1,
               /*KeepEmpty=*/false);

    inputs_attrs.emplace_back(input_name);

    const std::string new_inputs_attr_str =
        llvm::join(std::move(inputs_attrs), /*Separator=*/",");

    named_attr.setValue(StringAttr::get(ctx, new_inputs_attr_str));
  }

  main_func_op->setAttr("tf.entry_function",
                        DictionaryAttr::get(ctx, entry_func_attrs));
}

// Creates new arguments to the main function that corresponds to the source
// function's arguments. Returns the `IRMapping` that contains the
// relationship.
IRMapping CloneSrcFuncArgumentsToMainFunc(func::FuncOp src_func_op,
                                          func::FuncOp main_func_op) {
  IRMapping mapper{};

  for (auto [src_arg_idx, src_arg] :
       llvm::enumerate(src_func_op.getArguments())) {
    // No need to create a mapping when there is no usage - it will not affect
    // the cloning.
    if (src_arg.use_empty()) continue;

    const unsigned main_arg_idx = main_func_op.getNumArguments();

    const DictionaryAttr main_arg_attr =
        src_func_op.getArgAttrDict(src_arg_idx);

    main_func_op.insertArgument(main_arg_idx, src_arg.getType(), main_arg_attr,
                                src_arg.getLoc());

    const std::string new_input_name =
        absl::StrCat(GetInitializerType(src_func_op), "_", src_arg_idx, ":0");

    MaybeAddEntryFunctionInput(new_input_name, main_func_op);

    // During cloning, let it know that the source function's argument
    // corresponds to the main function's newly created argument when cloning
    // ops from src -> main.
    BlockArgument main_arg = main_func_op.getArgument(main_arg_idx);
    mapper.map(src_arg, main_arg);
  }

  return mapper;
}

// Copies ops from `src_func_op` to `main_body` except for the FetchOps. Returns
// the fetch values in the main GraphOp corresponding to the original fetch
// values from `src_func_op`. Returns an empty vector when `src_func_op` is
// empty. `main_func_op` must have a GraphOp.
SmallVector<Value> CopyOpsToMainFunction(func::FuncOp src_func_op,
                                         func::FuncOp main_func_op) {
  GraphOp src_graph_op = GetGraphOpFromFuncOp(src_func_op);
  if (!src_graph_op) {
    VLOG(1) << "Function " << src_func_op.getName().str()
            << " does not have a tf_executor::GraphOp. No ops are copied to "
               "the main function.";
    return {};
  }

  GraphOp main_graph_op = GetGraphOpFromFuncOp(main_func_op);

  FetchOp main_fetch_op = main_graph_op.GetFetch();
  const absl::Cleanup erase_main_fetch_op = [main_fetch_op]() mutable {
    main_fetch_op.erase();
  };

  // TODO(b/245473863): Handle when assets are actually used in the body.
  IRMapping mapper = CloneSrcFuncArgumentsToMainFunc(src_func_op, main_func_op);

  // Clones each op from src to main_body.
  Block& main_body = main_graph_op.GetBody();
  Block& src_body = src_graph_op.GetBody();
  for (Operation& op : src_body.without_terminator()) {
    main_body.push_back(op.clone(mapper));
  }

  // Relocate the main function's FetchOp at the last.
  main_body.push_back(main_fetch_op->clone(mapper));

  // Clone the source's FetchOp, but do not push to the main function's body.
  // The clone is only needed to identify the fetch operands.
  auto cloned_fetch_op = cast<FetchOp>(src_graph_op.GetFetch()->clone(mapper));
  const absl::Cleanup erase_cloned_fetch_op = [cloned_fetch_op]() mutable {
    cloned_fetch_op.erase();
  };

  return llvm::to_vector(cloned_fetch_op.getFetches());
}

// Creates a new `IslandOp` that wraps a `TF::NoOp`. The `IslandOp` has control
// dependencies to the values provided.
IslandOp CreateNoOpWithControlDependencies(
    const Location loc, GraphOp main_graph_op,
    const ArrayRef<Value> control_dependencies) {
  auto builder = OpBuilder::atBlockTerminator(&main_graph_op.GetBody());

  auto wrapper_island_op = builder.create<IslandOp>(
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
void AddFetchOperandToMain(GraphOp main_graph_op, const Value fetch_operand) {
  FetchOp old_fetch = main_graph_op.GetFetch();
  const absl::Cleanup erase_old_fetch = [old_fetch]() mutable {
    old_fetch.erase();
  };

  auto fetches = llvm::to_vector(old_fetch.getFetches());
  fetches.emplace_back(fetch_operand);

  auto builder = OpBuilder::atBlockTerminator(&main_graph_op.GetBody());
  builder.create<FetchOp>(main_graph_op.getLoc(), std::move(fetches));
}

// Creates a new Location for the initializer function. This creates a loc by
// attaching a to the initializer function's type so that it is identifiable.
Location CreateInitOpLoc(MLIRContext* ctx, func::FuncOp init_func_ops) {
  const std::string init_type = GetInitializerType(init_func_ops);
  const std::string name =
      absl::StrCat(init_type, "_", init_func_ops.getName().str());
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

  GraphOp main_graph_op = GetGraphOpFromFuncOp(main_func_op);
  if (!main_graph_op) return;

  SessionInitializerOp session_init_op = GetSessionInitializerOp(module_op);
  if (!session_init_op) return;

  // initializer_type -> init_func_op mapping.
  SymbolTable symbol_table{module_op};
  FailureOr<absl::flat_hash_map<std::string, func::FuncOp>> init_func_ops =
      GetInitFuncOps(session_init_op, symbol_table);
  if (failed(init_func_ops)) {
    module_op->emitError("Validation on initializer functions failed.");
    return signalPassFailure();
  } else if (init_func_ops->empty()) {
    VLOG(1) << "No initializer functions found.";
    return;
  }

  // Find the initializer functions and clone their ops to @main.
  for (auto& [init_type, init_op_func] : *init_func_ops) {
    const SmallVector<Value> init_op_fetches =
        CopyOpsToMainFunction(init_op_func, main_func_op);
    if (init_op_fetches.empty()) {
      VLOG(1) << "No fetch values exist from initializer functions.";
      return;
    }

    // Creates a NoOp that has control dependency to the initializer function
    // for non-variables.
    const Location init_op_loc = CreateInitOpLoc(ctx, init_op_func);
    IslandOp noop_wrapper_island_op = CreateNoOpWithControlDependencies(
        init_op_loc, main_graph_op,
        /*control_dependencies=*/init_op_fetches);

    AddFetchOperandToMain(
        main_graph_op,
        /*fetch_operand=*/noop_wrapper_island_op.getControl());

    symbol_table.erase(init_op_func);
  }

  // Empties the "initializers" attribute from the `SessionInitializerOp` since
  // all ops of the initializer ops are cloned into @main.
  session_init_op.setInitializersAttr(ArrayAttr::get(ctx, {}));
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
