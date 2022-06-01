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

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/macros.h"

namespace mlir {
namespace quant {
namespace {

constexpr char kEntryFunctionAttr[] = "tf.entry_function";
constexpr char kExportedNameAttr[] = "tf_saved_model.exported_names";
constexpr char kIndexPathAttr[] = "tf_saved_model.index_path";

// The ConvertMlirToGraphdef requires the provided input module to have a main
// function, which might not exist in case of multi-signature graphs. In that
// case, this pass will create a new main function, which calls signature
// functions.
class InsertMainFunctionPass
    : public PassWrapper<InsertMainFunctionPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertMainFunctionPass)

  explicit InsertMainFunctionPass() {}

  StringRef getArgument() const override { return "quant-add-main-function"; }

  StringRef getDescription() const override {
    return "Insert the main function to the module if it is missing.";
  }

  void runOnOperation() override;
};

// Checks if the module has a main function.
bool HasMainFunction(ModuleOp& module) {
  StringAttr main_func_id = StringAttr::get(module.getContext(), "main");
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.getName() == main_func_id) return true;
  }
  return false;
}

// Checks if a FuncOp is exported.
bool IsExported(func::FuncOp& op) {
  auto exported_names = op->getAttrOfType<ArrayAttr>(kExportedNameAttr);
  return exported_names && !exported_names.empty();
}

// Check if a function is an entry function.
bool IsEntryFunction(func::FuncOp& op) {
  return op->hasAttr(kEntryFunctionAttr);
}

// Sets a function to be private so it can be referred internally.
void SetFunctionPrivate(func::FuncOp& func) {
  func.setVisibility(SymbolTable::Visibility::Private);

  // The `tf_saved_model` attributes can only be appied to public functions.
  for (auto& attr : func->getAttrs()) {
    StringRef attr_name = attr.getName().getValue();
    if (attr_name.startswith("tf_saved_model.")) {
      func->removeAttr(attr_name);
    }
  }

  for (int i = 0; i < func.getNumArguments(); ++i) {
    for (auto& attr : func.getArgAttrs(i)) {
      const StringAttr& attr_name = attr.getName();
      if (attr_name.getValue().startswith("tf_saved_model.")) {
        func.removeArgAttr(i, attr_name);
      }
    }
  }
  for (int i = 0; i < func.getNumResults(); ++i) {
    for (auto& attr : func.getResultAttrs(i)) {
      const StringAttr& attr_name = attr.getName();
      if (attr_name.getValue().startswith("tf_saved_model.")) {
        func.removeResultAttr(i, attr_name);
      }
    }
  }
}

// Creates a main function which calls other exported functions.
bool CreateMainFunction(ModuleOp& module) {
  MLIRContext* context = module.getContext();
  OpBuilder builder(context);

  // Collects argument and result types.
  llvm::SmallVector<Location> arg_locs;
  llvm::SmallVector<Type> arg_types, result_types;
  std::vector<std::string> input_names, output_names;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isPrivate() || !IsExported(function)) continue;
    arg_types.append(function.getArgumentTypes().begin(),
                     function.getArgumentTypes().end());
    auto& return_op = function.getBody().getBlocks().front().back();
    result_types.append(return_op.getOperandTypes().begin(),
                        return_op.getOperandTypes().end());
    for (const auto& arg : function.getArguments()) {
      arg_locs.push_back(arg.getLoc());
    }

    // Collects input and output node names. These names are prefixed with the
    // signature key in SavedModel. They also contain the index suffix. Ex:
    // "<signature key>_<name>:0", where 0 is the index.
    if (auto tf_attrs =
            function->getAttrOfType<DictionaryAttr>(kEntryFunctionAttr)) {
      if (auto inputs_attr = tf_attrs.get("inputs")) {
        std::string inputs_attr_str =
            inputs_attr.cast<StringAttr>().getValue().str();
        std::vector<std::string> inputs_attr_vec =
            absl::StrSplit(inputs_attr_str, ',', absl::SkipEmpty());
        input_names.insert(input_names.end(), inputs_attr_vec.begin(),
                           inputs_attr_vec.end());
      }
      if (auto outputs_attr = tf_attrs.get("outputs")) {
        std::string outputs_attr_str =
            outputs_attr.cast<StringAttr>().getValue().str();
        std::vector<std::string> outputs_attr_vec =
            absl::StrSplit(outputs_attr_str, ',', absl::SkipEmpty());
        output_names.insert(output_names.end(), outputs_attr_vec.begin(),
                            outputs_attr_vec.end());
      }
    }
  }

  // Creates a new main function.
  auto func_type = FunctionType::get(context, arg_types, result_types);
  auto main_func =
      builder.create<func::FuncOp>(module.getLoc(), "main", func_type);
  builder.createBlock(&main_func.getBody(), main_func.begin(), arg_types,
                      arg_locs);
  SmallVector<NamedAttribute> func_attrs;
  func_attrs.push_back(
      {StringAttr::get(context, "inputs"),
       StringAttr::get(context, absl::StrJoin(input_names, ","))});
  func_attrs.push_back(
      {StringAttr::get(context, "outputs"),
       StringAttr::get(context, absl::StrJoin(output_names, ","))});
  auto dictAttr = DictionaryAttr::get(context, func_attrs);
  main_func->setAttr(StringAttr::get(context, kEntryFunctionAttr), dictAttr);
  main_func->setAttr(kExportedNameAttr, builder.getStrArrayAttr({"main"}));

  if (input_names.size() != main_func.getNumArguments() ||
      output_names.size() != main_func.getNumResults()) {
    module.emitError()
        << "Number of inputs and outputs in the tf.entry_function attribute "
           "mismatched. [Input] Expected: "
        << input_names.size() << ", got: " << main_func.getNumArguments()
        << ". [Output] Expected: " << output_names.size()
        << ", got: " << main_func.getNumResults();
    return false;
  }

  int numArgs = main_func.getNumArguments();
  for (int i = 0; i < numArgs; ++i) {
    main_func.setArgAttr(
        i, kIndexPathAttr,
        mlir::ArrayAttr::get(context,
                             {mlir::StringAttr::get(context, input_names[i])}));
  }

  int numResults = main_func.getNumResults();
  for (int i = 0; i < numResults; ++i) {
    main_func.setResultAttr(
        i, kIndexPathAttr,
        mlir::ArrayAttr::get(
            context, {mlir::StringAttr::get(context, output_names[i])}));
  }

  // Creates PartitionedCall ops to call exported functions.
  auto guard = OpBuilder::InsertionGuard(builder);
  int arg_idx = 0;
  int result_idx = 0;
  llvm::SmallVector<Value> returning_values;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isPrivate() || !IsExported(function) ||
        !IsEntryFunction(function)) {
      continue;
    }

    llvm::ArrayRef<BlockArgument> new_args = llvm::makeArrayRef(
        main_func.getArguments().begin() + arg_idx, function.getNumArguments());
    arg_idx += function.getNumArguments();
    llvm::ArrayRef<Type> new_types = llvm::makeArrayRef(
        result_types.begin() + result_idx, function.getNumResults());
    result_idx += function.getNumResults();

    auto call_op = builder.create<TF::PartitionedCallOp>(
        module.getLoc(), new_types, new_args,
        SymbolRefAttr::get(context, function.getSymName()),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    returning_values.append(call_op.getResults().begin(),
                            call_op.getResults().end());
    SetFunctionPrivate(function);
  }
  builder.create<mlir::func::ReturnOp>(main_func.getBody().getLoc(),
                                       returning_values);

  // Adds the new function to symbol table.
  SymbolTable symbol_table(module);
  symbol_table.insert(main_func);
  return true;
}

void InsertMainFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!HasMainFunction(module)) {
    if (!CreateMainFunction(module)) {
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateInsertMainFunctionPass() {
  return std::make_unique<InsertMainFunctionPass>();
}

static PassRegistration<InsertMainFunctionPass> pass([] {
  return CreateInsertMainFunctionPass();
});

}  // namespace quant
}  // namespace mlir
