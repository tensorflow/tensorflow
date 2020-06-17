/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.h"

#include <utility>

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/ViewOpGraph.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

Status HandleInputOutputArraysWithModule(const toco::ModelFlags& model_flags,
                                         mlir::OwningModuleRef* module) {
  mlir::FuncOp entry_function = nullptr;
  for (auto func : module->get().getOps<mlir::FuncOp>()) {
    if (auto tf_attrs =
            func.getAttrOfType<mlir::DictionaryAttr>("tf.entry_function")) {
      // TODO(jaesung): There could be multiple entry functions. Let's handle
      // such cases if there are any needs for that.
      if (entry_function != nullptr) {
        return errors::InvalidArgument(
            "There should be only one tf.entry_function");
      }
      entry_function = func;
    }
  }
  if (entry_function == nullptr) {
    return errors::InvalidArgument("no tf.entry_function found");
  }

  // Get the list of input Op names from the function attribute.
  mlir::DictionaryAttr tf_attrs =
      entry_function.getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  llvm::SmallVector<llvm::StringRef, 4> function_input_names;
  function_input_names.reserve(model_flags.input_arrays().size());
  auto input_attr = tf_attrs.get("inputs");
  if (!input_attr) {
    return errors::InvalidArgument("no inputs attribute found");
  }
  auto input_names = input_attr.cast<mlir::StringAttr>().getValue();
  input_names.split(function_input_names, ",");
  if (function_input_names.size() != model_flags.input_arrays().size()) {
    return errors::InvalidArgument(
        "input array size mismatch: got ", function_input_names.size(),
        ", expected: ", model_flags.input_arrays().size());
  }
  llvm::StringSet<> function_input_names_set;
  function_input_names_set.insert(function_input_names.begin(),
                                  function_input_names.end());
  for (const auto& input_array : model_flags.input_arrays()) {
    if (function_input_names_set.count(input_array.name()) == 0) {
      return errors::InvalidArgument("input array name (", input_array.name(),
                                     ") does not exist in the given graph");
    }
  }

  // Get the list of output Op names from the function attribute.
  llvm::SmallVector<llvm::StringRef, 4> function_output_names;
  function_output_names.reserve(model_flags.output_arrays().size());
  auto output_attr = tf_attrs.get("outputs");
  if (!output_attr) {
    return errors::InvalidArgument("no outputs attribute found");
  }
  auto output_names = output_attr.cast<mlir::StringAttr>().getValue();
  output_names.split(function_output_names, ",");
  if (function_output_names.size() != model_flags.output_arrays().size()) {
    return errors::InvalidArgument(
        "output array size mismatch: got ", function_output_names.size(),
        ", expected: ", model_flags.output_arrays().size());
  }
  llvm::StringSet<> function_output_names_set;
  function_output_names_set.insert(function_output_names.begin(),
                                   function_output_names.end());
  for (const auto& output_array : model_flags.output_arrays()) {
    if (function_output_names_set.count(output_array) == 0) {
      return errors::InvalidArgument("output array name (", output_array,
                                     ") does not exist in the given graph");
    }
  }
  return Status::OK();
}

Status ConvertSavedModelToTFLiteFlatBuffer(
    const toco::ModelFlags& model_flags, const toco::TocoFlags& toco_flags,
    string* result) {
  mlir::MLIRContext context;
  mlir::TFL::QuantizationSpecs quant_specs;

  // Parse input arrays.
  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<std::vector<int>> node_shapes;
  std::vector<llvm::Optional<double>> node_mins;
  std::vector<llvm::Optional<double>> node_maxs;

  // Populate quantization specs.
  TF_RETURN_IF_ERROR(internal::PopulateQuantizationSpecs(
      model_flags, toco_flags, &quant_specs, &node_names, &node_dtypes,
      &node_shapes, &node_mins, &node_maxs));

  internal::WarningUnusedFlags(model_flags, toco_flags);

  // Register all custom ops, including user-specified custom ops.
  TF_RETURN_IF_ERROR(internal::RegisterAllCustomOps(toco_flags));

  auto& saved_model_tags = model_flags.saved_model_tags();
  auto& saved_model_exported_names = model_flags.saved_model_exported_names();
  std::unordered_set<std::string> tags(saved_model_tags.begin(),
                                       saved_model_tags.end());
  auto exported_names_in_vector = std::vector<std::string>(
      saved_model_exported_names.begin(), saved_model_exported_names.end());
  absl::Span<std::string> exported_names(exported_names_in_vector);

  if (exported_names.size() != 1) {
    return errors::Unimplemented("Only support a single exported name.");
  }

  TF_ASSIGN_OR_RETURN(auto module,
                      ImportSavedModel(model_flags.saved_model_dir(),
                                       model_flags.saved_model_version(), tags,
                                       exported_names, &context));

  if (!model_flags.input_arrays().empty() ||
      !model_flags.output_arrays().empty()) {
    TF_RETURN_IF_ERROR(HandleInputOutputArraysWithModule(model_flags, &module));
  }

  mlir::TFL::PassConfig pass_config(quant_specs);
  bool emit_builtin_tflite_ops = !toco_flags.force_select_tf_ops();
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.lower_tensor_list_ops = true;

  auto status = internal::ConvertMLIRToTFLiteFlatBuffer(
      toco_flags, std::move(module), pass_config, result);
  return status;
}

}  // namespace tensorflow
