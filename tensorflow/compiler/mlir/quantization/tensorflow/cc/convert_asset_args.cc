/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/convert_asset_args.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir::quant {
namespace {

using ::mlir::tf_saved_model::AssetOp;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::mlir::tf_saved_model::LookupBoundInputOfType;
using ::tensorflow::AssetFileDef;

// Given argument attributes `arg_attrs`, returns a new set of argument
// attributes where the "tf_saved_model.bound_input" attribute has been replaced
// with the "tf_saved_model.index_path" attribute. `index_path` is the element
// of the index path attribute.
SmallVector<NamedAttribute> ReplaceBoundInputAttrWithIndexPathAttr(
    const ArrayRef<NamedAttribute> arg_attrs, const StringRef index_path,
    Builder& builder) {
  // Keep all other attributes except the tf_saved_model.bound_input attribute,
  // as we are replacing it with tf_saved_model.index_path.
  SmallVector<NamedAttribute> new_arg_attrs;
  for (auto arg_attr : arg_attrs) {
    if (arg_attr.getName() == "tf_saved_model.bound_input") continue;
    new_arg_attrs.emplace_back(arg_attr);
  }

  const NamedAttribute index_path_attr(
      builder.getStringAttr(kTfSavedModelIndexPathAttr),
      builder.getStrArrayAttr({index_path}));

  new_arg_attrs.emplace_back(index_path_attr);
  return new_arg_attrs;
}

// Strips the "assets/" directory prefix, if `filename` begins with it. The
// SavedModel loader attaches the prefix for you during loading.
StringRef MaybeStripAssetDirectoryPrefix(const StringRef filename) {
  if (filename.find("assets/") == 0) {
    return filename.drop_front(7);
  } else {
    return filename;
  }
}

AssetFileDef CreateAssetFileDef(const StringRef filename,
                                const StringRef tensor_name) {
  AssetFileDef asset_file_def{};
  asset_file_def.set_filename(MaybeStripAssetDirectoryPrefix(filename).str());

  tensorflow::TensorInfo tensor_info{};
  tensor_info.set_name(tensor_name.str());
  *asset_file_def.mutable_tensor_info() = tensor_info;

  return asset_file_def;
}

// Returns a list of "tf.entry_function" attribute's "inputs" comma-split
// values.
//
// Example: if `func_op` has attribute `tf.entry_function = {inputs =
// "arg0:0,arg1:0"}`, then this function returns `{"arg0:0", "arg1:0"}`.
SmallVector<StringRef> GetEntryFunctionInputs(func::FuncOp func_op) {
  auto entry_function_attr =
      func_op->getAttrOfType<DictionaryAttr>("tf.entry_function");

  SmallVector<StringRef> inputs;
  mlir::dyn_cast_or_null<StringAttr>(entry_function_attr.get("inputs"))
      .strref()
      .split(inputs, /*Separator=*/",");

  return inputs;
}

void ConvertMainArgAttrs(func::FuncOp main_func_op, const int arg_idx,
                         const StringRef index_path) {
  const ArrayRef<NamedAttribute> arg_attrs =
      main_func_op.getArgAttrDict(arg_idx).getValue();

  Builder builder(main_func_op.getContext());
  SmallVector<NamedAttribute> new_arg_attrs =
      ReplaceBoundInputAttrWithIndexPathAttr(arg_attrs, index_path, builder);

  main_func_op.setArgAttrs(arg_idx, new_arg_attrs);
}

}  // namespace

FailureOr<SmallVector<AssetFileDef>> ConvertAssetArgs(ModuleOp module_op) {
  func::FuncOp main_func_op = FindMainFuncOp(module_op);
  if (!main_func_op) return failure();

  SmallVector<StringRef> input_names = GetEntryFunctionInputs(main_func_op);
  SymbolTable symbol_table(module_op);
  SmallVector<AssetFileDef> asset_file_defs;

  for (BlockArgument argument : main_func_op.getArguments()) {
    const int arg_idx = argument.getArgNumber();
    auto asset_op =
        LookupBoundInputOfType<AssetOp>(main_func_op, arg_idx, symbol_table);
    if (!asset_op) continue;

    const StringRef input_name = input_names[arg_idx];
    ConvertMainArgAttrs(main_func_op, arg_idx, /*index_path=*/input_name);

    // This assumes that the final tensor name in `GraphDef` is equal to
    // `input_name`.
    asset_file_defs.emplace_back(CreateAssetFileDef(
        asset_op.getFilenameAttr(), /*tensor_name=*/input_name));
  }

  return asset_file_defs;
}

}  // namespace mlir::quant
