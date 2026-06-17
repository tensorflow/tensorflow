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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_REPRESENTATIVE_DATASET_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_REPRESENTATIVE_DATASET_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace stablehlo::quantization {

// Translates a set of `RepresentativeDatsetConfig` to signature key ->
// `RepresentativeDatasetFile` mapping. This is useful when using
// `RepresentativeDatasetConfig`s at places that accept the legacy
// `RepresentativeDatasetFile` mapping.
// Returns a non-OK status when there is a duplicate signature key among
// `representative_dataset_configs`.
absl::StatusOr<absl::flat_hash_map<
    std::string, tensorflow::quantization::RepresentativeDatasetFile>>
CreateRepresentativeDatasetFileMap(absl::Span<const RepresentativeDatasetConfig>
                                       representative_dataset_configs);

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_REPRESENTATIVE_DATASET_H_
