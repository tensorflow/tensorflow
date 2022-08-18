/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace quantization {
namespace internal {

absl::StatusOr<GraphDef> QuantizeQatModel(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags, absl::string_view quant_opts_serialized);

// Apply post-training dynamic range quantization to the model.
absl::StatusOr<GraphDef> QuantizePtqDynamicRange(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags, absl::string_view quant_opts_serialized);

absl::StatusOr<GraphDef> QuantizePtqModelPreCalibration(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags);

absl::StatusOr<GraphDef> QuantizePtqModelPostCalibration(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags, absl::string_view quant_opts_serialized);

}  // namespace internal
}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_H_
