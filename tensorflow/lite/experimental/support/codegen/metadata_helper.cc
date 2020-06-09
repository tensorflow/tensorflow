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

#include "tensorflow/lite/experimental/support/codegen/metadata_helper.h"

#include "tensorflow/lite/experimental/support/codegen/utils.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace support {
namespace codegen {

constexpr char BUFFER_KEY[] = "TFLITE_METADATA";
const ModelMetadata* GetMetadataFromModel(const Model* model) {
  if (model == nullptr || model->metadata() == nullptr) {
    return nullptr;
  }
  for (auto i = 0; i < model->metadata()->size(); i++) {
    const auto* name = model->metadata()->Get(i)->name();
    if (name != nullptr && name->str() == BUFFER_KEY) {
      const auto buffer_index = model->metadata()->Get(i)->buffer();
      if (model->buffers() == nullptr ||
          model->buffers()->size() <= buffer_index) {
        continue;
      }
      const auto* buffer_vec = model->buffers()->Get(buffer_index)->data();
      if (buffer_vec == nullptr || buffer_vec->data() == nullptr) {
        continue;
      }
      return GetModelMetadata(buffer_vec->data());
    }
  }
  return nullptr;
}

int FindAssociatedFile(const TensorMetadata* metadata,
                       const AssociatedFileType file_type,
                       const std::string& tensor_identifier,
                       ErrorReporter* err) {
  int result = -1;
  if (metadata->associated_files() == nullptr ||
      metadata->associated_files()->size() == 0) {
    return result;
  }
  for (int i = 0; i < metadata->associated_files()->size(); i++) {
    const auto* file_metadata = metadata->associated_files()->Get(i);
    if (file_metadata->type() == file_type) {
      if (result >= 0) {
        err->Warning(
            "Multiple associated file of type %d found on tensor %s. Only the "
            "first one will be used.",
            file_type, tensor_identifier.c_str());
        continue;
      }
      result = i;
    }
  }
  return result;
}

int FindNormalizationUnit(const TensorMetadata* metadata,
                          const std::string& tensor_identifier,
                          ErrorReporter* err) {
  int result = -1;
  if (metadata->process_units() == nullptr ||
      metadata->process_units()->size() == 0) {
    return result;
  }
  for (int i = 0; i < metadata->process_units()->size(); i++) {
    const auto* process_uint = metadata->process_units()->Get(i);
    if (process_uint->options_type() ==
        ProcessUnitOptions_NormalizationOptions) {
      if (result >= 0) {
        err->Warning(
            "Multiple normalization unit found in tensor %s. Only the first "
            "one will be effective.",
            tensor_identifier.c_str());
        continue;
      }
      result = i;
    }
  }
  return result;
}

}  // namespace codegen
}  // namespace support
}  // namespace tflite
