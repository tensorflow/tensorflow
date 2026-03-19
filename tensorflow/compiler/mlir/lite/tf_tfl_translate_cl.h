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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_TRANSLATE_CL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_TRANSLATE_CL_H_

// This file contains command-line options aimed to provide the parameters
// required by the TensorFlow Graph(Def) to TF Lite Flatbuffer conversion. It is
// only intended to be included by binaries.

#include <string>

#include "llvm/Support/CommandLine.h"

// The commandline options are defined in LLVM style, so the caller should
// use llvm::InitLLVM to initialize the options.
//
// Please see the implementation file for documentation of details of these
// options.
// TODO(jpienaar): Revise the command line option parsing here.
extern llvm::cl::opt<std::string> input_file_name;
extern llvm::cl::opt<std::string> output_file_name;
extern llvm::cl::opt<bool> use_splatted_constant;
extern llvm::cl::opt<bool> input_mlir;
extern llvm::cl::opt<bool> output_mlir;
extern llvm::cl::list<std::string> custom_opdefs;
extern llvm::cl::opt<bool> emit_quant_adaptor_ops;
extern llvm::cl::opt<std::string> quant_stats_file_name;
extern llvm::cl::opt<bool> convert_tf_while_to_tfl_while;
extern llvm::cl::opt<std::string> select_user_tf_ops;
extern llvm::cl::opt<bool> allow_all_select_tf_ops;
extern llvm::cl::opt<bool> unfold_batchmatmul;
extern llvm::cl::opt<bool> unfold_large_splat_constant;
extern llvm::cl::opt<bool> guarantee_all_funcs_one_use;
extern llvm::cl::opt<bool> enable_dynamic_update_slice;
extern llvm::cl::opt<bool> preserve_assert_op;
extern llvm::cl::opt<bool> legalize_custom_tensor_list_ops;
extern llvm::cl::opt<bool> reduce_type_precision;
extern llvm::cl::opt<std::string> input_arrays;
extern llvm::cl::opt<std::string> input_dtypes;
extern llvm::cl::opt<std::string> input_shapes;
extern llvm::cl::opt<std::string> output_arrays;
extern llvm::cl::opt<std::string> control_output_arrays;
extern llvm::cl::opt<std::string> inference_type;
extern llvm::cl::opt<std::string> min_values;
extern llvm::cl::opt<std::string> max_values;
extern llvm::cl::opt<std::string> debug_info_file;
extern llvm::cl::opt<bool> upgrade_legacy;
extern llvm::cl::opt<bool> enable_shape_inference;

// Import saved model.
extern llvm::cl::opt<bool> import_saved_model_object_graph;
extern llvm::cl::opt<bool> import_saved_model_signature_defs;
extern llvm::cl::opt<std::string> saved_model_tags;
extern llvm::cl::opt<std::string> saved_model_exported_names;

// Import HLO.
enum HloImportType { proto, hlotxt, mlir_text };

extern llvm::cl::opt<bool> import_hlo;
extern llvm::cl::opt<HloImportType> hlo_import_type;

// enable_hlo_to_tf_conversion and disable_hlo_to_tfl_conversion are used to
// control the HLO to TF and HLO to TFLite conversion while debugging an
// input_mlir. The default value of enable_hlo_to_tf_conversion is false, and
// the default value of disable_hlo_to_tfl_conversion is true.
extern llvm::cl::opt<bool> enable_hlo_to_tf_conversion;
extern llvm::cl::opt<bool> disable_hlo_to_tfl_conversion;

// quantization related flags
extern llvm::cl::opt<bool> post_training_quantization;

// TF to stablehlo pass flags
extern llvm::cl::opt<bool> enable_stablehlo_conversion;

// Whether to enable the attempt to directly lower composites into tflite ops or
// not.
extern llvm::cl::opt<bool> enable_composite_direct_lowering;

// The source model type
extern llvm::cl::opt<std::string> model_origin_framework;

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_TRANSLATE_CL_H_
