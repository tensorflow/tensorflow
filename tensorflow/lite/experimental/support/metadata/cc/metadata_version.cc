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

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace metadata {
namespace {

// Members that are added to the metadata schema after the initial version
// of 1.0.0.
enum class SchemaMembers {
  kAssociatedFileTypeVocabulary = 0,
};

// Helper class to compare semantic versions in terms of three integers, major,
// minor, and patch.
class Version {
 public:
  explicit Version(int major, int minor = 0, int patch = 0)
      : version_({major, minor, patch}) {}

  explicit Version(const std::string& version) {
    const std::vector<std::string> vec = absl::StrSplit(version, '.');
    // The version string should always be less than four numbers.
    TFLITE_DCHECK(vec.size() <= kElementNumber && !vec.empty());
    version_[0] = std::stoi(vec[0]);
    version_[1] = vec.size() > 1 ? std::stoi(vec[1]) : 0;
    version_[2] = vec.size() > 2 ? std::stoi(vec[2]) : 0;
  }

  // Compares two semantic version numbers.
  //
  // Example results when comparing two versions strings:
  //   "1.9" precedes "1.14";
  //   "1.14" precedes "1.14.1";
  //   "1.14" and "1.14.0" are equal.
  //
  // Returns the value 0 if the two versions are equal; a value less than 0 if
  // *this precedes v; a value greater than 0 if v precedes *this.
  int Compare(const Version& v) {
    for (int i = 0; i < kElementNumber; ++i) {
      if (version_[i] != v.version_[i]) {
        return version_[i] < v.version_[i] ? -1 : 1;
      }
    }
    return 0;
  }

  // Converts version_ into a version string.
  std::string ToString() { return absl::StrJoin(version_, "."); }

 private:
  static constexpr int kElementNumber = 3;
  std::array<int, kElementNumber> version_;
};

Version GetMemberVersion(SchemaMembers member) {
  switch (member) {
    case SchemaMembers::kAssociatedFileTypeVocabulary:
      return Version(1, 0, 1);
    default:
      TFLITE_LOG(FATAL) << "Unsupported schema member: "
                        << static_cast<int>(member);
  }
}

// Updates min_version if it precedes the new_version.
inline void UpdateMinimumVersion(const Version& new_version,
                                 Version* min_version) {
  if (min_version->Compare(new_version) < 0) {
    *min_version = new_version;
  }
}

void UpdateMinimumVersionForAssociatedFile(
    const tflite::AssociatedFile* associated_file, Version* min_version) {
  if (associated_file == nullptr) return;

  if (associated_file->type() == AssociatedFileType_VOCABULARY) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kAssociatedFileTypeVocabulary),
        min_version);
  }
}

void UpdateMinimumVersionForAssociatedFileArray(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::AssociatedFile>>*
        associated_files,
    Version* min_version) {
  if (associated_files == nullptr) return;

  for (int i = 0; i < associated_files->size(); ++i) {
    UpdateMinimumVersionForAssociatedFile(associated_files->Get(i),
                                          min_version);
  }
}

void UpdateMinimumVersionForTensorMetadata(
    const tflite::TensorMetadata* tensor_metadata, Version* min_version) {
  if (tensor_metadata == nullptr) return;

  // Checks the associated_files field.
  UpdateMinimumVersionForAssociatedFileArray(
      tensor_metadata->associated_files(), min_version);
}

void UpdateMinimumVersionForTensorMetadataArray(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
        tensor_metadata_array,
    Version* min_version) {
  if (tensor_metadata_array == nullptr) return;

  for (int i = 0; i < tensor_metadata_array->size(); ++i) {
    UpdateMinimumVersionForTensorMetadata(tensor_metadata_array->Get(i),
                                          min_version);
  }
}

void UpdateMinimumVersionForSubGraphMetadata(
    const tflite::SubGraphMetadata* subgraph_metadata, Version* min_version) {
  if (subgraph_metadata == nullptr) return;

  // Checks in the input/output metadata arrays.
  UpdateMinimumVersionForTensorMetadataArray(
      subgraph_metadata->input_tensor_metadata(), min_version);
  UpdateMinimumVersionForTensorMetadataArray(
      subgraph_metadata->output_tensor_metadata(), min_version);

  // Checks the associated_files field.
  UpdateMinimumVersionForAssociatedFileArray(
      subgraph_metadata->associated_files(), min_version);
}

void UpdateMinimumVersionForModelMetadata(
    const tflite::ModelMetadata& model_metadata, Version* min_version) {
  // Checks the subgraph_metadata field.
  if (model_metadata.subgraph_metadata() != nullptr) {
    for (int i = 0; i < model_metadata.subgraph_metadata()->size(); ++i) {
      UpdateMinimumVersionForSubGraphMetadata(
          model_metadata.subgraph_metadata()->Get(i), min_version);
    }
  }

  // Checks the associated_files field.
  UpdateMinimumVersionForAssociatedFileArray(model_metadata.associated_files(),
                                             min_version);
}

}  // namespace

TfLiteStatus GetMinimumMetadataParserVersion(const uint8_t* buffer_data,
                                             size_t buffer_size,
                                             std::string* min_version_str) {
  flatbuffers::Verifier verifier =
      flatbuffers::Verifier(buffer_data, buffer_size);
  if (!tflite::VerifyModelMetadataBuffer(verifier)) {
    TFLITE_LOG(ERROR) << "The model metadata is not a valid FlatBuffer buffer.";
    return kTfLiteError;
  }

  static constexpr char kDefaultVersion[] = "1.0.0";
  Version min_version = Version(kDefaultVersion);

  // Checks if any member declared after 1.0.0 (such as those in
  // SchemaMembers) exists, and updates min_version accordingly. The minimum
  // metadata parser version will be the largest version number of all fields
  // that has been added to a metadata flatbuffer
  const tflite::ModelMetadata* model_metadata = GetModelMetadata(buffer_data);

  // All tables in the metadata schema should have their dedicated
  // UpdateMinimumVersionFor**() methods, respectively. We'll gradually add
  // these methods when new fields show up in later schema versions.
  //
  // UpdateMinimumVersionFor<Foo>() takes a const pointer of Foo. The pointer
  // can be a nullptr if Foo is not populated into the corresponding table of
  // the Flatbuffer object. In this case, UpdateMinimumVersionFor<Foo>() will be
  // skipped. An exception is UpdateMinimumVersionForModelMetadata(), where
  // ModelMetadata is the root table, and it won't be null.
  UpdateMinimumVersionForModelMetadata(*model_metadata, &min_version);

  *min_version_str = min_version.ToString();
  return kTfLiteOk;
}

}  // namespace metadata
}  // namespace tflite
