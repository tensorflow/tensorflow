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

#include "tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h"

#include "llvm/Support/CommandLine.h"

using llvm::cl::opt;

// TODO(jpienaar): Revise the command line option parsing here.
// NOLINTNEXTLINE
opt<std::string> input_file_name(llvm::cl::Positional,
                                 llvm::cl::desc("<input file>"),
                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
opt<bool> import_saved_model_object_graph(
    "savedmodel-objectgraph-to-mlir",
    llvm::cl::desc("Import a saved model to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
opt<bool> import_saved_model_signature_defs(
    "savedmodel-signaturedefs-to-mlir",
    llvm::cl::desc("Import a saved model V1 to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
opt<std::string> saved_model_tags(
    "tf-savedmodel-tags",
    llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    llvm::cl::init("serve"));

// NOLINTNEXTLINE
opt<std::string> saved_model_exported_names(
    "tf-savedmodel-exported-names",
    llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> output_file_name("o", llvm::cl::desc("<output file>"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init("-"));
// NOLINTNEXTLINE
opt<bool> use_splatted_constant(
    "use-splatted-constant",
    llvm::cl::desc(
        "Replace constants with randomly generated splatted tensors"),
    llvm::cl::init(false), llvm::cl::Hidden);
// NOLINTNEXTLINE
opt<bool> input_mlir(
    "input-mlir",
    llvm::cl::desc("Take input TensorFlow model in textual MLIR instead of "
                   "GraphDef format"),
    llvm::cl::init(false), llvm::cl::Hidden);
// NOLINTNEXTLINE
opt<bool> output_mlir(
    "output-mlir",
    llvm::cl::desc(
        "Output MLIR rather than FlatBuffer for the generated TFLite model"),
    llvm::cl::init(false));
// NOLINTNEXTLINE
opt<bool> allow_all_select_tf_ops(
    "allow-all-select-tf-ops",
    llvm::cl::desc("Allow automatic pass through of TF ops (outside the flex "
                   "allowlist) as select Tensorflow ops"),
    llvm::cl::init(false));

// The following approach allows injecting opdefs in addition
// to those that are already part of the global TF registry  to be linked in
// prior to importing the graph. The primary goal is for support of custom ops.
// This is not intended to be a general solution for custom ops for the future
// but mainly for supporting older models like mobilenet_ssd. More appropriate
// mechanisms, such as op hints or using functions to represent composable ops
// like https://github.com/tensorflow/community/pull/113 should be encouraged
// going forward.
// NOLINTNEXTLINE
llvm::cl::list<std::string> custom_opdefs(
    "tf-custom-opdefs", llvm::cl::desc("List of custom opdefs when importing "
                                       "graphdef"));

// Quantize and Dequantize ops pair can be optionally emitted before and after
// the quantized model as the adaptors to receive and produce floating point
// type data with the quantized model. Set this to `false` if the model input is
// integer types.
// NOLINTNEXTLINE
opt<bool> emit_quant_adaptor_ops(
    "emit-quant-adaptor-ops",
    llvm::cl::desc(
        "Emit Quantize/Dequantize before and after the generated TFLite model"),
    llvm::cl::init(false));

// The path to a quantization stats file to specify value ranges for some of the
// tensors with known names.
// NOLINTNEXTLINE
opt<std::string> quant_stats_file_name("quant-stats",
                                       llvm::cl::desc("<stats file>"),
                                       llvm::cl::value_desc("filename"),
                                       llvm::cl::init(""));

// A list of comma separated TF operators which are created by the user.
// This must be used with `-emit-select-tf-ops=true`.
// NOLINTNEXTLINE
opt<std::string> select_user_tf_ops(
    "select-user-tf-ops",
    llvm::cl::desc(
        "<list of custom tf ops created by the user (comma separated)>"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<bool> unfold_batchmatmul(
    "unfold_batchmatmul",
    llvm::cl::desc(
        "Whether to unfold TF BatchMatMul to a set of TFL FullyConnected ops."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> unfold_large_splat_constant(
    "unfold-large-splat-constant",
    llvm::cl::desc("Whether to unfold large splat constant tensors to reduce "
                   "the generated model size."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> guarantee_all_funcs_one_use(
    "guarantee-all-funcs-one-use",
    llvm::cl::desc(
        "Whether to clone functions to ensure each function has a single use."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> enable_dynamic_update_slice(
    "enable-dynamic-update-slice",
    llvm::cl::desc("Whether to enable dynamic update slice op to convert "
                   "TensorListSetItem op."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> import_hlo("import-hlo",
                     llvm::cl::desc("Whether the input file is hlo file."),
                     llvm::cl::init(false));

// NOLINTNEXTLINE
opt<HloImportType> hlo_import_type(
    "hlo-import-type", llvm::cl::desc("The file type of the hlo."),
    llvm::cl::values(clEnumVal(proto, "Import hlo in proto binary format"),
                     clEnumVal(hlotxt, "Import hlo in hlotxt format"),
                     clEnumVal(mlir_text, "Import hlo in mlir_text format")));

// NOLINTNEXTLINE
opt<bool> enable_hlo_to_tf_conversion(
    "enable-hlo-to-tf-conversion",
    llvm::cl::desc("Whether to enable the hlo to tf ops conversion."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> disable_hlo_to_tfl_conversion(
    "disable-hlo-to-tfl-conversion",
    llvm::cl::desc("Whether to disable the hlo to tfl ops conversion."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> preserve_assert_op(
    "preserve-assert-op",
    llvm::cl::desc("Preserve AssertOp during tfl legalization."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> enable_stablehlo_conversion(
    "enable-stablehlo-conversion",
    llvm::cl::desc("Enable converting TF to Stablehlo."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> post_training_quantization(
    "post-training-quantization",
    llvm::cl::desc("Enable post_training_quantization."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> legalize_custom_tensor_list_ops(
    "legalize-custom-tensor-list-ops",
    llvm::cl::desc("Convert \"tf.TensorList*\" ops to \"tfl.custom_op\""
                   "if they can all be supported."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> reduce_type_precision(
    "reduce-type-precision",
    llvm::cl::desc("Convert tensors to a lower precision if all values are "
                   "within the reduced precision range. This could have side "
                   "effects triggered by downstream packing algorithms."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> enable_composite_direct_lowering(
    "enable-composite-direct-lowering",
    llvm::cl::desc("Whether to enable the attempt to directly lower composites "
                   "into tflite ops or not."),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<std::string> model_origin_framework(
    "model-origin-framework",
    llvm::cl::desc("The source model type: PYTORCH, JAX, TENSORFLOW, etc."),
    llvm::cl::init("UNSET"));

// NOLINTNEXTLINE
opt<std::string> input_arrays(
    "tf-input-arrays", llvm::cl::desc("Input tensor names, separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> input_dtypes(
    "tf-input-data-types",
    llvm::cl::desc("(Optional) Input tensor data types, separated by ','. Use "
                   "'' if a single data type is skipped. The data type from "
                   "the import graph is used if it is skipped."),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> input_shapes(
    "tf-input-shapes",
    llvm::cl::desc(
        "Input tensor shapes. Shapes for different tensors are separated by "
        "':', and dimension sizes for the same tensor are separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> output_arrays(
    "tf-output-arrays", llvm::cl::desc("Output tensor names, separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> control_output_arrays(
    "tf-control-output-arrays",
    llvm::cl::desc("Control output node names, separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> inference_type(
    "tf-inference-type",
    llvm::cl::desc(
        "Sets the type of real-number arrays in the output file. Only allows "
        "float and quantized types"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> min_values(
    "tf-input-min-values",
    llvm::cl::desc(
        "Sets the lower bound of the input data. Separated by ','; Each entry "
        "in the list should match an entry in -tf-input-arrays. This is "
        "used when -tf-inference-type is a quantized type."),
    llvm::cl::Optional, llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> max_values(
    "tf-input-max-values",
    llvm::cl::desc(
        "Sets the upper bound of the input data. Separated by ','; Each entry "
        "in the list should match an entry in -tf-input-arrays. This is "
        "used when -tf-inference-type is a quantized type."),
    llvm::cl::Optional, llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> debug_info_file(
    "tf-debug-info",
    llvm::cl::desc("Path to the debug info file of the input graph def"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<bool> enable_shape_inference(
    "tf-enable-shape-inference-on-import",
    llvm::cl::desc("Enable shape inference on import (temporary)"),
    llvm::cl::init(false));
