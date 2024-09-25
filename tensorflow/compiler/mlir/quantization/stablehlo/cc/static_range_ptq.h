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

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir::quant::stablehlo {

// Component for static-range post-training quantization (PTQ).
// TODO: b/320607042 - Add tests in python level.
class StaticRangePtqComponent : public Component {
 public:
  // Name of this component. Used for debugging purposes.
  static constexpr absl::string_view kName = "quant_static_range_ptq";

  // Constructs `StaticRangePtqComponent` by creating three sub-components:
  // `PreCalibrationComponent`, `CalibrationComponent`, and
  // `PostCalibrationComponent`. These are stored in `sub_components_` in
  // sequence. All arguments except `ctx` is used to initialize
  // `CalibrationComponent`. For detailed explanation of each argument, see the
  // comment of `CalibrationComponent`'s constructor.
  StaticRangePtqComponent(
      absl::Nonnull<MLIRContext*> ctx,
      absl::Nonnull<const tensorflow::quantization::PyFunctionLibrary*>
          py_function_library,
      absl::string_view src_saved_model_path,
      std::vector<std::string> signature_keys,
      std::unordered_set<std::string> tags,
      absl::flat_hash_map<std::string, tensorflow::SignatureDef>
          signature_def_map,
      absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases);

  // Runs the static-range post-training quantization (PTQ) on `module_op`.
  absl::StatusOr<ModuleOp> Run(
      ModuleOp module_op,
      const ::stablehlo::quantization::QuantizationConfig& config) override;

 private:
  // A non-owning `MLIRContext`. This `MLIRContext` should exceed the lifetime
  // of `StaticRangePtqComponent`.
  absl::Nonnull<MLIRContext*> ctx_;
  // This component consists of three sub-components, `PreCalibrationComponent`,
  // `CalibrationComponent`, and `PostCalibrationComponent`.
  std::array<std::unique_ptr<Component>, 3> sub_components_;
};

// Runs static-range post-training quantization (PTQ) on a SavedModel at
// `src_saved_model_path` and saves the resulting model to
// `dst_saved_model_path`.
//
// `quantization_config` configures the quantization behavior for the
// static-range PTQ.
//
// `signature_keys` specify the signatures that correspond to functions to be
// quantized. `signature_def_map` connects the signature keys to
// `SignatureDef`s.
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
    const tensorflow::quantization::PyFunctionLibrary& py_function_library);
// LINT.ThenChange(../python/pywrap_quantization.cc:static_range_ptq)

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_STATIC_RANGE_PTQ_H_
