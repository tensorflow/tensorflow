/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_

#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Load a TF model from a GraphDef definition or a TF control flow dialect MLIR
// source into a MLIR module. If `input_mlir` is true, load from a MLIR source
// file; otherwise, load from a GraphDef.
// Setting prune_unused_nodes to true, would prune unreachable nodes if
// output_arrays is specified.
stream_executor::port::StatusOr<mlir::OwningModuleRef>
LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    absl::string_view debug_info_file, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view inference_type,
    absl::string_view min_values, absl::string_view max_values,
    bool prune_unused_nodes, llvm::SourceMgr* source_mgr,
    mlir::MLIRContext* context);

// Quantization passess will run only when the user specifies a quantized type
// in the `-tf-inference-type` flag, which is converted to the function
// attribute "tf.quantize" by the importer module.
// TODO(fengliuai): switch to the cmd flag once the flags are moved to this
// file with main method.
bool ShouldRunQuantizePasses(mlir::ModuleOp m);

// Add the MLIR passes that convert TF control flow dialect to TF Lite dialect
// to a MLIR `pass_manager`. These passes first raise the control flow in the TF
// control flow dialect, decode the constant tensors, and then legalize the
// module to TF Lite dialect with some optimizations afterwards.
// If `emit_builtin_tflite_ops` is true, TF Lite legalization passes will be
// added, which produces TF Lite ops. If `run_quantize` is true, quantization
// passes will be added. If `emit_quant_adaptor_ops` is true, Quantize and
// Dequantize ops are added to the inputs and outputs of the quantized model.
// If `lower_tensor_list_ops` is true, tensorlist ops will be lowered to basic
// TF ops before legalization to TF Lite dialect.
void AddTFToTFLConversionPasses(bool emit_builtin_tflite_ops, bool run_quantize,
                                bool emit_quant_adaptor_ops,
                                bool lower_tensor_list_ops,
                                mlir::PassManager* pass_manager);

// Taking a MLIR module in TF control flow dialect and a set of parameters,
// applies a set of passes to convert the module to TF Lite dialect and
// serializes the result to a string. Depending on an attribute in the module
// main function, Quantization is applied. If `export_to_mlir` is true, the
// result is exported in MLIR text format, otherwise exported in flat buffer.
Status ConvertTFControlFlowToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, bool emit_builtin_tflite_ops,
    bool emit_select_tf_ops, bool emit_custom_ops, bool emit_quant_adaptor_ops,
    bool lower_tensor_list_ops, std::string* result);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
