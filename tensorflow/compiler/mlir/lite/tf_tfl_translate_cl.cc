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

using llvm::cl::opt;

// TODO(jpienaar): Revise the command line option parsing here.
// NOLINTNEXTLINE
opt<std::string> input_file_name(llvm::cl::Positional,
                                 llvm::cl::desc("<input file>"),
                                 llvm::cl::init("-"));
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

// NOLINTNEXTLINE
opt<bool> inline_functions(
    "inline",
    llvm::cl::desc("Inline function calls within the main function "
                   "before legalization to TFLite."),
    llvm::cl::init(true));
