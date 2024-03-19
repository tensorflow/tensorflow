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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_DEBUGGER_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_DEBUGGER_H_

#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace stablehlo::quantization {

// Enables debugging on `exported_model` by updating the `DumpTensor` ops.
//
// Saves the current model to `debugger_options.unquantized_dump_model_path()`
// if the debugger type is `DEBUGGER_TYPE_WHOLE_MODEL`. This is required because
// in whole-model debugging mode the `DumpTensor` ops for the unquantized
// tensors are only inserted in the unquantized model whereas `DumpTensor` ops
// for the quantized tensors are only inserted in the quantized model. Both
// models are required to be able to dump both quantized and unquantized tensors
// and compare them offline.
void EnableDebugging(
    tensorflow::quantization::ExportedModel& exported_model,
    const stablehlo::quantization::DebuggerConfig& debugger_config,
    const tensorflow::quantization::PyFunctionLibrary& py_function_library,
    absl::string_view src_saved_model_path,
    const std::unordered_set<std::string>& tags,
    const absl::flat_hash_map<std::string, tensorflow::SignatureDef>&
        signature_def_map);

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_DEBUGGER_H_
