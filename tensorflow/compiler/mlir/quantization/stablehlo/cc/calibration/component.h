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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_COMPONENT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_COMPONENT_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
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

// Performs post-calibration graph transformation as part of post-training
// static-range quantization.
//
// The resulting `ModuleOp` contains quantized StableHLO ops serialized in
// `TF::XlaCallModuleOp`s. They are quantized using the statistics collected
// after the calibration step, corresponding to each `TF::CustomAggregatorOp`s
// in the input module op.
//
// TODO: b/320607042 - Add tests for this component on the python layer.
class CalibrationComponent : public Component {
 public:
  // Name of the post-training quantization post-calibration step. Used for
  // debugging purposes.
  static constexpr absl::string_view kName = "quant_ptq_calibration";

  // `CalibrationComponent` ctor with necessary information required to run
  // calibration on a `ModuleOp`. Meta information like `function_aliases`,
  // `tags`, `signature_def_map`, and `signature_keys` are required to properly
  // save and load the module_op to and from SavedModel.
  // `representative_dataset_file_map` contains information about the
  // calibration dataset.
  CalibrationComponent(
      absl::Nonnull<MLIRContext*> ctx,
      absl::Nonnull<const tensorflow::quantization::PyFunctionLibrary*>
          py_function_lib,
      absl::string_view src_saved_model_path,
      absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases,
      std::unordered_set<std::string> tags,
      absl::flat_hash_map<std::string, tensorflow::SignatureDef>
          signature_def_map,
      std::vector<std::string> signature_keys);

  // Runs calibration on `module_op` and returns a calibrated ModuleOp with
  // calibrated statistics embedded.
  absl::StatusOr<ModuleOp> Run(
      ModuleOp module_op,
      const ::stablehlo::quantization::QuantizationConfig& config) override;

 private:
  // Exports `module_op` to SavedModel at `dst_saved_model_path`. This is used
  // to export the pre-calibrated `module_op` to SavedModel so that the
  // calibration process can use it to load and run the graph with the
  // representative dataset.
  absl::StatusOr<tensorflow::quantization::ExportedModel> ExportToSavedModel(
      ModuleOp module_op, absl::string_view dst_saved_model_path);

  // Imports the SavedModel at `calibrated_saved_model_path` to `ModuleOp` after
  // running calibration.
  absl::StatusOr<ModuleOp> ImportCalibratedSavedModel(
      absl::string_view calibrated_saved_model_path);

  absl::Nonnull<MLIRContext*> ctx_;

  // Contains function implementations from the python layer. Should be injected
  // from the python level using pybind11.
  absl::Nonnull<const tensorflow::quantization::PyFunctionLibrary*>
      py_function_lib_;

  // Path to the pre-calibrated SavedModel.
  std::string src_saved_model_path_;

  // Function alias mapping for pre-calibrated SavedModel. Used to preserve
  // aliased functions.
  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases_;

  // Tags to identify the MetaGraphDef to load from a SavedModel.
  const std::unordered_set<std::string> tags_;

  const absl::flat_hash_map<std::string, tensorflow::SignatureDef>
      signature_def_map_;

  // Signature keys to identify the functions to load & quantize.
  const std::vector<std::string> signature_keys_;
};

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_COMPONENT_H_
