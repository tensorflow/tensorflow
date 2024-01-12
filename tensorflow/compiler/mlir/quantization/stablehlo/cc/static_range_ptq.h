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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_STATIC_RANGE_PTQ_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_STATIC_RANGE_PTQ_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir::quant::stablehlo {

// Runs static-range post-training quantization (PTQ) on a SavedModel at
// `src_saved_model_path` and saves the resulting model to
// `dst_saved_model_path`.
//
// `quantization_config` configures the quantization behavior for the
// static-range PTQ.
//
// `signature_keys` specify the signatures that correspond to functions to be
// quantized. `signature_def_map` connects the signature keys to
// `SignatureDef`s. `function_aliases` maps actual function names to the
// function aliases, as defined by the
// `MetaGraphDef::MetaInfoDef::function_aliases` from the input SavedModel.
//
// Returns a non-OK status when the quantization is not successful.
// LINT.IfChange
absl::Status QuantizeStaticRangePtq(
    absl::string_view src_saved_model_path,
    absl::string_view dst_saved_model_path,
    const ::stablehlo::quantization::QuantizationConfig& quantization_config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, tensorflow::SignatureDef>&
        signature_def_map,
    const absl::flat_hash_map<std::string, std::string>& function_aliases,
    const tensorflow::quantization::PyFunctionLibrary& py_function_library,
    const absl::flat_hash_map<
        std::string, tensorflow::quantization::RepresentativeDatasetFile>&
        representative_dataset_file_map);
// LINT.ThenChange(../python/pywrap_quantization.cc:static_range_ptq)

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_STATIC_RANGE_PTQ_H_
