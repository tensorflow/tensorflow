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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_METADATA_HELPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_METADATA_HELPER_H_

#include <string>

#include "tensorflow/lite/experimental/support/codegen/utils.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace support {
namespace codegen {

/// Parses a ModelMetadata out from a Model. The returned ModelMetadata's
/// lifetime is scoped by the model.
/// Returns nullptr if we cannot find any metadata.
const ModelMetadata* GetMetadataFromModel(const Model* model);

/// Finds an associated file from a TensorMetadata of certain type. If there're
/// multiple files meet the criteria, only the first one is used. If there's no
/// file meets the criteria, -1 will be returned.
int FindAssociatedFile(const TensorMetadata* metadata,
                       const AssociatedFileType file_type,
                       const std::string& tensor_identifier,
                       ErrorReporter* err);

/// Find the first normalization unit. If none, return -1.
int FindNormalizationUnit(const TensorMetadata* metadata,
                          const std::string& tensor_identifier,
                          ErrorReporter* err);

}  // namespace codegen
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_METADATA_HELPER_H_
