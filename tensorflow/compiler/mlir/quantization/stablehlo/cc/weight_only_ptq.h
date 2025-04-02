/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_WEIGHT_ONLY_PTQ_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_WEIGHT_ONLY_PTQ_H_

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir::quant::stablehlo {

// Performs int8 weight-only quantization on dot_general ops.
//
// The resulting `ModuleOp` contains quantized StableHLO ops serialized in
// `TF::XlaCallModuleOp`s. They are quantized using the weight constants, not
// relying on calibration.
class WeightOnlyPtqComponent : public Component {
 public:
  // Used for debugging purposes.
  static constexpr absl::string_view kName = "quant_ptq_weight_only";

  explicit WeightOnlyPtqComponent(MLIRContext* /*absl_nonnull*/ ctx);

  absl::StatusOr<ModuleOp> Run(
      ModuleOp module_op,
      const ::stablehlo::quantization::QuantizationConfig& config) override;

 private:
  MLIRContext* /*absl_nonnull*/ ctx_;
};

// Runs weight-only quantization on a SavedModel at
// `src_saved_model_path` and saves the resulting model to
// `dst_saved_model_path`.
//
// `quantization_config` configures the quantization behavior for the
// weight-only quantization.
//
// `signature_keys` specify the signatures that correspond to functions to be
// quantized. `signature_def_map` connects the signature keys to
// `SignatureDef`s.
//
// Returns a non-OK status when the quantization is not successful.
// LINT.IfChange
absl::Status QuantizeWeightOnlyPtq(
    absl::string_view src_saved_model_path,
    absl::string_view dst_saved_model_path,
    ::stablehlo::quantization::QuantizationConfig quantization_config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, tensorflow::SignatureDef>&
        signature_def_map,
    const tensorflow::quantization::PyFunctionLibrary& py_function_library);
// LINT.ThenChange(../python/pywrap_quantization.cc:weight_only_ptq)

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_WEIGHT_ONLY_PTQ_H_
