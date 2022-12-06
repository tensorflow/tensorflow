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
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerTypeAttr;
using ::mlir::tf_saved_model::SessionInitializerOp;

constexpr absl::string_view kDefaultConstName = "const";

class UnfreezeConstantsPass
    : public PassWrapper<UnfreezeConstantsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnfreezeConstantsPass)

  explicit UnfreezeConstantsPass() = default;

  StringRef getArgument() const override { return "quant-unfreeze-constants"; }

  StringRef getDescription() const override {
    return "Unfreeze large constants.";
  }

  void runOnOperation() override;

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect,
                    tf_saved_model::TensorFlowSavedModelDialect>();
  }
};

// Adds the symbol to the "initializers" attribute of the session_initializer
// op.
void AddSymbolToInitializersAttr(SessionInitializerOp session_init_op,
                                 FlatSymbolRefAttr symbol) {
  const auto prev_initializers = session_init_op.getInitializersAttr();
  llvm::SmallVector<Attribute> initializers_attrs{prev_initializers.begin(),
                                                  prev_initializers.end()};
  initializers_attrs.emplace_back(symbol);

  session_init_op.setInitializersAttr(
      ArrayAttr::get(session_init_op.getContext(), initializers_attrs));
}

// Create the initializer function right after the session_initializer op.
// Returns the newly created initializer function. The initializer function's
// initializer_type is set to "restore_op" since it essentially serves as a
// variable restoration function.
func::FuncOp CreateInitializerFunc(SymbolTable& symbol_table,
                                   SessionInitializerOp session_init_op) {
  OpBuilder builder{session_init_op.getContext()};
  builder.setInsertionPointAfter(session_init_op);

  const Location loc = builder.getUnknownLoc();
  const auto func_type = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});

  auto init_func = builder.create<func::FuncOp>(
      loc, /*sym_name=*/"init_func_restore_op", func_type);
  builder.createBlock(&init_func.getBody(), /*insertPt=*/init_func.begin(),
                      /*arg_types=*/{}, /*arg_locs=*/{});

  init_func->setAttr(kTfSavedModelExportedNamesAttr,
                     builder.getStrArrayAttr(
                         {"tf_saved_model.session_initializer_restore_op"}));
  init_func->setAttr(
      kTfSavedModelInitializerTypeAttr,
      builder.getStringAttr(kTfSavedModelInitializerRestoreType));

  builder.setInsertionPointToStart(&init_func.front());
  builder.create<func::ReturnOp>(loc, /*operands=*/ValueRange{});

  symbol_table.insert(init_func);

  AddSymbolToInitializersAttr(
      session_init_op, FlatSymbolRefAttr::get(init_func.getSymNameAttr()));

  return init_func;
}

// Returns true if the initializer function's tf_saved_model.initializer_type
// matches `initializer_type`.
bool IsInitializerType(func::FuncOp init_func_op, StringRef initializer_type) {
  auto init_type =
      init_func_op->getAttrOfType<StringAttr>(kTfSavedModelInitializerTypeAttr);
  return init_type && init_type == initializer_type;
}

// Returns the initializer function whose tf_saved_model.initializer_type
// matches `initializer_type`. Creates and returns a new initializer function
// iff such FuncOp is not found. The newly created initializer function's symbol
// will be added to the symbol table and session_initializer op's "intializer"
// attribute.
func::FuncOp GetOrCreateSessionInitializerFunc(
    SymbolTable& symbol_table, SessionInitializerOp session_init_op,
    StringRef initializer_type) {
  for (const auto init_sym :
       session_init_op.getInitializers().getAsValueRange<FlatSymbolRefAttr>()) {
    auto init_func_op = symbol_table.lookup<func::FuncOp>(init_sym);
    if (!init_func_op) continue;

    if (IsInitializerType(init_func_op, kTfSavedModelInitializerRestoreType)) {
      return init_func_op;
    }
  }

  // Create a new initializer function if the init function is not found.
  return CreateInitializerFunc(symbol_table, session_init_op);
}

// Returns the session_initializer op in the module if exists. Otherwise,
// creates a new session_initializer op and returns it.
SessionInitializerOp GetOrCreateSessionInitializerOp(ModuleOp module_op) {
  SessionInitializerOp session_init_op =
      tf_saved_model::GetSessionInitializerOp(module_op);

  // Create one if it doesn't exist.
  if (!session_init_op) {
    OpBuilder builder{&module_op.getBodyRegion()};

    session_init_op = builder.create<SessionInitializerOp>(
        module_op.getLoc(), /*initializers=*/builder.getArrayAttr({}));
  }

  return session_init_op;
}

// Retrieve the ConstOp's name from its loc. Returns "const" if a name cannot be
// produced from its loc.
std::string GetConstOpName(TF::ConstOp const_op) {
  if (const std::string name = GetNameFromLoc(const_op.getLoc());
      !name.empty()) {
    return name;
  }

  return std::string(kDefaultConstName);
}

// Collects the ConstOps to unfreeze.
std::vector<TF::ConstOp> GetTargetConstOps(ModuleOp module_op) {
  std::vector<TF::ConstOp> target_const_ops{};

  // TODO(b/254636388): Lift the assumption that there are no intializer
  // functions and avoid converting ConstOps inside initializer functions.
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    auto const_ops = func_op.getOps<TF::ConstOp>();
    target_const_ops.insert(target_const_ops.end(), const_ops.begin(),
                            const_ops.end());
  }

  return target_const_ops;
}

// Replaces every uses of ConstOps in `target_const_ops` to VarHandleOp ->
// ReadVariableOp patterns. The ConstOps are not erased. Returns the ConstOp ->
// shared_name mapping. The shared_name is the shared name of the corresponding
// VarHandleOp.
llvm::MapVector<TF::ConstOp, std::string> ReplaceConstOpUsesWithVariableReads(
    llvm::ArrayRef<TF::ConstOp> target_const_ops) {
  llvm::MapVector<TF::ConstOp, std::string> const_op_name_map{};

  absl::flat_hash_map<std::string, int> name_counts{};
  for (auto const_op : target_const_ops) {
    OpBuilder builder{const_op};

    // TODO(b/254635554): Hoist VarHandleOp to the outermost function and pass
    // down as arguments to avoid relying on shared variables.
    const std::string name = GetConstOpName(const_op);
    const int cnt = name_counts[name]++;

    // Creates a unique name by appending its occurrence count.
    const auto shared_name = absl::StrCat(name, "_", cnt);
    const_op_name_map[const_op] = shared_name;

    // Creates a VarHandleOp -> ReadVariableOp pair for each ConstOp.
    const auto resource_type = RankedTensorType::get(
        /*shape=*/{}, /*elementType=*/TF::ResourceType::get(
            /*subtypes=*/llvm::ArrayRef<TensorType>{const_op.getType()},
            builder.getContext()));
    auto var_handle_op =
        builder.create<TF::VarHandleOp>(const_op.getLoc(),
                                        /*resource=*/resource_type,
                                        /*container=*/"", shared_name);

    auto read_variable_op = builder.create<TF::ReadVariableOp>(
        const_op.getLoc(), const_op.getType(), var_handle_op);

    // Replace each usage of ConstOp with the corresponding ReadVariableOp.
    const_op.getResult().replaceAllUsesWith(read_variable_op);
  }

  return const_op_name_map;
}

// Inside `session_init_func`, creates AssignVariableOps(VarHandleOp, ConstOp)
// for each VarHandleOp that replaces a ConstOp. The `session_init_func` will
// essentially behave like restore_op for the newly created VarHandleOps whose
// shared names are the values of `const_op_name_map`.
void CreateAssignVariableOps(
    llvm::MapVector<TF::ConstOp, std::string>& const_op_name_map,
    func::FuncOp session_init_func) {
  OpBuilder builder{&session_init_func.getBody()};

  for (auto& [const_op, shared_name] : const_op_name_map) {
    const auto element_type = TF::ResourceType::get(
        /*subtypes=*/llvm::ArrayRef<TensorType>{const_op.getType()},
        builder.getContext());

    const auto ranked_tensor_type = RankedTensorType::get(
        /*shape=*/{}, /*elementType=*/element_type);
    auto var_handle_op =
        builder.create<TF::VarHandleOp>(const_op.getLoc(),
                                        /*resource=*/ranked_tensor_type,
                                        /*container=*/"", shared_name);

    // Assign the ConstOp to each VarHandleOp. These will be used to save the
    // variable values to the checkpoint.
    auto const_op_copy =
        builder.create<TF::ConstOp>(const_op.getLoc(), const_op.getValue());

    builder.create<TF::AssignVariableOp>(const_op.getLoc(),
                                         /*resource=*/var_handle_op,
                                         /*value=*/const_op_copy.getOutput());
  }
}

void UnfreezeConstantsPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  // Find the ConstOps to "unfreeze" into VarHandleOps.
  const std::vector<TF::ConstOp> target_const_ops =
      GetTargetConstOps(module_op);
  if (target_const_ops.empty()) {
    VLOG(1) << "No ConstOps found. UnfreezeConstantsPass is a no-op.";
    return;
  }

  SessionInitializerOp session_init_op =
      GetOrCreateSessionInitializerOp(module_op);

  SymbolTable symbol_table{module_op};
  func::FuncOp session_init_func = GetOrCreateSessionInitializerFunc(
      symbol_table, session_init_op, kTfSavedModelInitializerRestoreType);

  // Replace each usage of ConstOp to a VarHandleOp -> ReadVariableOp pattern.
  llvm::MapVector<TF::ConstOp, std::string> const_op_name_map =
      ReplaceConstOpUsesWithVariableReads(target_const_ops);

  // In the session initializer function, assign the const op's values to the
  // corresponding VarHandleOps.
  CreateAssignVariableOps(const_op_name_map, session_init_func);

  // Erase the ConstOps that are replaced by VarHandleOps.
  absl::c_for_each(target_const_ops, [](auto const_op) { const_op.erase(); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateUnfreezeConstantsPass() {
  return std::make_unique<UnfreezeConstantsPass>();
}

static PassRegistration<UnfreezeConstantsPass> pass([] {
  return CreateUnfreezeConstantsPass();
});

}  // namespace quant
}  // namespace mlir
