/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/support/metadata/cc/metadata_version.h"

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace metadata {

TfLiteStatus GetMinimumMetadataParserVersion(const uint8_t* buffer_data,
                                             size_t buffer_size,
                                             std::string* min_version) {
  flatbuffers::Verifier verifier =
      flatbuffers::Verifier(buffer_data, buffer_size);
  if (!tflite::VerifyModelMetadataBuffer(verifier)) {
    TFLITE_LOG(ERROR) << "The model metadata is not a valid FlatBuffer buffer.";
    return kTfLiteError;
  }

  // Returns the version as the initial default one, "1.0.0", because it is the
  // first version ever for metadata_schema.fbs.
  //
  // Later, when new fields are added to the schema, we'll update the logic of
  // getting the minimum metadata parser version. To be more specific, we'll
  // have a table that records the new fields and the versions of the schema
  // they are added to. And the minimum metadata parser version will be the
  // largest version number of all fields that has been added to a metadata
  // flatbuffer.
  // TODO(b/156539454): replace the hardcoded version with template + genrule.
  static constexpr char kDefaultVersion[] = "1.0.0";
  *min_version = kDefaultVersion;
  return kTfLiteOk;
}

}  // namespace metadata
}  // namespace tflite
