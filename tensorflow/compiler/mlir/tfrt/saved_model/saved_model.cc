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

#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tensorflow {
namespace {

llvm::StringRef ProcessIndexPath(mlir::ArrayAttr index_path) {
  if (index_path.size() == 1 && index_path[0].isa<mlir::StringAttr>()) {
    // TODO(chky): Support cases where index_path is not a single string.
    return index_path[0].cast<mlir::StringAttr>().getValue();
  }
  return "";
}

}  // namespace

void MapFunctionSignaturesFromTFSavedModelMLIR(
    mlir::ModuleOp module,
    llvm::function_ref<void(
        llvm::StringRef func_name,
        llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>>
            input_names_and_devices,
        llvm::ArrayRef<llvm::StringRef> output_names,
        llvm::ArrayRef<mlir::tf_saved_model::GlobalTensorOp> global_tensors)>
        map_fn) {
  // Create global_tensors for each functions.
  mlir::SymbolTable symbol_table(module);
  module.walk([&symbol_table, map_fn](mlir::FuncOp func) {
    // Use the exported name as the function name, and skip non-exported
    // functions.
    auto func_names = mlir::tf_saved_model::GetExportedNames(func);
    if (func_names.empty()) return;

    // Here we walk through each arguments and find out the input/output names,
    // and input devices, variables used by this function.
    llvm::SmallVector<std::pair<llvm::StringRef, llvm::StringRef>, 4>
        input_names_and_devices;
    llvm::SmallVector<mlir::tf_saved_model::GlobalTensorOp, 4> global_tensors;
    for (unsigned i = 0, e = func.getNumArguments(); i != e; ++i) {
      if (auto input_index_path = func.getArgAttrOfType<mlir::ArrayAttr>(
              i, "tf_saved_model.index_path")) {
        std::pair<llvm::StringRef, llvm::StringRef> name_and_device;
        name_and_device.first = ProcessIndexPath(input_index_path);
        if (auto input_device =
                func.getArgAttrOfType<mlir::StringAttr>(i, "tf.device")) {
          name_and_device.second = input_device.getValue();
        }
        input_names_and_devices.push_back(name_and_device);
      }
      if (auto variable =
              mlir::tf_saved_model::LookupBoundInput(func, i, symbol_table)) {
        global_tensors.push_back(variable);
      }
    }

    llvm::SmallVector<llvm::StringRef, 4> output_names;
    for (unsigned i = 0, e = func.getNumResults(); i != e; ++i) {
      if (auto output_index_path = func.getResultAttrOfType<mlir::ArrayAttr>(
              i, "tf_saved_model.index_path")) {
        output_names.push_back(ProcessIndexPath(output_index_path));
      }
    }

    for (auto func_name : func_names)
      map_fn(func_name, input_names_and_devices, output_names, global_tensors);
  });
}

Status CompileTFSavedModelMLIRToBEF(const TFRTSavedModelCompileOptions& options,
                                    mlir::ModuleOp module,
                                    tfrt::AlignedBuffer<8>* bef_buffer) {
  VLOG(1) << "TF Dialect: " << tensorflow::MlirModuleToString(module);

  // Lower MLIR TF Dialect to MLIR TFRT CoreRT dialect.
  mlir::PassManager pm(module.getContext());

  tensorflow::CoreRTPipelineOptions pass_options;
  if (!options.default_device.empty()) {
    pass_options.default_device = options.default_device;
  }
  if (!options.force_data_format.empty()) {
    pass_options.force_data_format = options.force_data_format;
  }
  pass_options.enable_optimizer = options.enable_optimizer;
  tensorflow::CreateTFExecutorToCoreRTPipeline(pm, pass_options);

  if (mlir::failed(pm.run(module)))
    return tensorflow::errors::Internal(
        "failed to lower TF Dialect to CoreRT dialect.");

  VLOG(1) << "TFRT Dialect: " << tensorflow::MlirModuleToString(module);

  auto bef =
      tfrt::ConvertMLIRToBEF(module, /* disable_optional_sections = */ true);
  if (bef.empty())
    return tensorflow::errors::Internal("failed to convert MLIR to BEF.");

  assert(bef_buffer);
  bef_buffer->assign(bef.begin(), bef.end());

  return Status::OK();
}

}  // namespace tensorflow
