/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"

#include "absl/strings/str_split.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime

namespace tensorflow {
namespace {

using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;

llvm::StringRef ProcessIndexPath(mlir::ArrayAttr index_path) {
  if (index_path.size() == 1 && index_path[0].isa<mlir::StringAttr>()) {
    // TODO(chky): Support cases where index_path is not a single string.
    return index_path[0].cast<mlir::StringAttr>().getValue();
  }
  return "";
}

StatusOr<std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>>
ProcessTensorSpec(mlir::TensorType type) {
  tensorflow::DataType dtype;
  TF_RETURN_IF_ERROR(
      ConvertScalarTypeToDataType(type.getElementType(), &dtype));

  if (!type.hasRank())
    return std::make_pair(dtype, tensorflow::PartialTensorShape());

  auto shape = type.getShape();
  llvm::SmallVector<int64_t, 4> dims;
  dims.assign(shape.begin(), shape.end());
  return std::make_pair(dtype, tensorflow::PartialTensorShape(dims));
}

}  // namespace

Status MapFunctionSignaturesFromTFSavedModelMLIR(
    mlir::ModuleOp module,
    llvm::function_ref<void(const TFRTSavedModelSignatureInfo&)> map_fn) {
  // Create bound inputs for each functions.
  mlir::SymbolTable symbol_table(module);
  tensorflow::Status status = OkStatus();
  module.walk([&symbol_table, map_fn, &status](mlir::func::FuncOp func) {
    // Use the exported name as the function name, and skip non-exported
    // functions.
    auto func_names = mlir::tf_saved_model::GetExportedNames(func);
    if (func_names.empty()) return mlir::WalkResult::advance();

    auto func_type = func.getFunctionType();

    // Here we walk through each arguments and find out the input/output names,
    // and input devices, variables used by this function.
    llvm::SmallVector<llvm::StringRef, 4> input_names;
    llvm::SmallVector<
        std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>, 4>
        input_specs;
    llvm::SmallVector<llvm::StringRef, 4> input_devices;
    llvm::SmallVector<mlir::Operation*, 4> bound_inputs;
    for (unsigned i = 0, e = func.getNumArguments(); i != e; ++i) {
      if (auto input_index_path = func.getArgAttrOfType<mlir::ArrayAttr>(
              i, kTfSavedModelIndexPathAttr)) {
        input_names.push_back(ProcessIndexPath(input_index_path));
        auto statusor_spec =
            ProcessTensorSpec(func_type.getInput(i).cast<mlir::TensorType>());
        if (!statusor_spec.ok()) {
          status = std::move(statusor_spec).status();
          return mlir::WalkResult::interrupt();
        }
        input_specs.push_back(std::move(statusor_spec).value());
        if (auto input_device =
                func.getArgAttrOfType<mlir::StringAttr>(i, "tf.device")) {
          input_devices.push_back(input_device.getValue());
        } else {
          input_devices.push_back("");
        }
      }
      if (auto* bound_input =
              mlir::tf_saved_model::LookupBoundInput(func, i, symbol_table)) {
        bound_inputs.push_back(bound_input);
      }
    }

    llvm::SmallVector<llvm::StringRef, 4> output_names;
    llvm::SmallVector<
        std::pair<tensorflow::DataType, tensorflow::PartialTensorShape>, 4>
        output_specs;
    for (unsigned i = 0, e = func.getNumResults(); i != e; ++i) {
      if (auto output_index_path = func.getResultAttrOfType<mlir::ArrayAttr>(
              i, kTfSavedModelIndexPathAttr)) {
        output_names.push_back(ProcessIndexPath(output_index_path));
        auto statusor_spec =
            ProcessTensorSpec(func_type.getResult(i).cast<mlir::TensorType>());
        if (!statusor_spec.ok()) {
          status = std::move(statusor_spec).status();
          return mlir::WalkResult::interrupt();
        }
        output_specs.push_back(std::move(statusor_spec).value());
      }
    }

    for (auto func_name : func_names) {
      TFRTSavedModelSignatureInfo sig_info;
      sig_info.func_name = func_name;
      sig_info.input_names = input_names;
      sig_info.input_specs = input_specs;
      sig_info.input_devices = input_devices;
      sig_info.output_names = output_names;
      sig_info.output_specs = output_specs;
      sig_info.bound_inputs = bound_inputs;
      map_fn(sig_info);
    }

    return mlir::WalkResult::advance();
  });

  return status;
}

}  // namespace tensorflow
