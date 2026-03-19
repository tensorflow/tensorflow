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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/representative_dataset.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace stablehlo::quantization {

using ::tensorflow::quantization::RepresentativeDatasetFile;

absl::StatusOr<absl::flat_hash_map<std::string, RepresentativeDatasetFile>>
CreateRepresentativeDatasetFileMap(absl::Span<const RepresentativeDatasetConfig>
                                       representative_dataset_configs) {
  absl::flat_hash_map<std::string, RepresentativeDatasetFile>
      repr_dataset_file_map{};

  for (const RepresentativeDatasetConfig& dataset_config :
       representative_dataset_configs) {
    RepresentativeDatasetFile repr_dataset_file;

    repr_dataset_file.set_tfrecord_file_path(dataset_config.tf_record().path());
    // If the signature_key has not been explicitly specified, use the default
    // value of "serving_default".
    const std::string signature_key = dataset_config.has_signature_key()
                                          ? dataset_config.signature_key()
                                          : "serving_default";
    if (repr_dataset_file_map.contains(signature_key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("RepresentativeDatasetConfig should not contain "
                       "duplicate signature key: ",
                       signature_key));
    }
    repr_dataset_file_map[signature_key] = std::move(repr_dataset_file);
  }

  return repr_dataset_file_map;
}

}  // namespace stablehlo::quantization
