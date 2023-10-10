/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/py_wrapper/tac_wrapper.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_module.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"

namespace tflite {
namespace {
std::unique_ptr<mlir::TFL::tac::TacImporter> CreateTfLiteImporter(
    const std::string input_file_name) {
  mlir::TFL::tac::TfLiteImporter::Options options;
  options.file_name = input_file_name;
  options.input_mlir = false;
  return std::make_unique<mlir::TFL::tac::TfLiteImporter>(options);
}

std::unique_ptr<mlir::TFL::tac::TacExporter> CreateTfLiteExporter(
    const std::string output_file_name,
    const std::vector<std::string>& target_hardware_backends) {
  mlir::TFL::tac::TfLiteExporter::Options options;
  options.output_mlir = false;
  options.output_file_name = output_file_name;
  options.export_runtime_metadata = false;
  options.target_hardware_backends = target_hardware_backends;
  return std::make_unique<mlir::TFL::tac::TfLiteExporter>(options);
}
}  // namespace

// Run target-aware-conversion for the given tflite model with the given device
// specs.
// Warning: The API is experimental and subject to changes.
bool run_tac(const std::string& model_file_path,
             const std::vector<std::string>& device_specs,
             const std::string& model_output_path) {
  mlir::TFL::tac::TacModule::Options options;
  options.hardware_backends = device_specs;
  options.enable_inliner = true;
  options.legalize_to_tflite_ops = true;
  mlir::TFL::tac::TacModule tac_module(options);
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  tac_module.RegisterExtraDialects(registry);
  tac_module.SetImporter(CreateTfLiteImporter(model_file_path));
  tac_module.SetExporter(
      CreateTfLiteExporter(model_output_path, options.hardware_backends));
  return tac_module.Run().ok();
}

}  // namespace tflite
