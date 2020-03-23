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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_CODE_GENERATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_CODE_GENERATOR_H_

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "tensorflow/lite/experimental/support/codegen/utils.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace support {
namespace codegen {

struct GenerationResult {
  struct File {
    std::string path;
    std::string content;
  };
  std::vector<File> files;
};

/// Defines language-independent codegen strategies, like class naming, .etc.
/// Should not be used directly.
class CodeGenerator {
 public:
  CodeGenerator();

  using TensorMetadataList =
      typename flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>>;

  virtual ~CodeGenerator() {}

  // Strategies.
  /// Names all the IO tensors. It's useful when they don't have names, or the
  /// names have conflicts. We have to name every tensor for code generation.
  // TODO(b/141225157): Add reserved keywords check.
  static std::pair<std::vector<std::string>, std::vector<std::string>>
  NameInputsAndOutputs(const TensorMetadataList* inputs,
                       const TensorMetadataList* outputs);

  /// Loads a metadata for code generation.
  /// Returns false if the metadata is not good for generation.
  static bool VerifyMetadata(const ModelMetadata* metadata, ErrorReporter* err);

 protected:
  /// Converts a name into a valid form. Rules:
  /// - lower all letters.
  /// - replace all non alphabet nor numeric characters with underscores.
  /// - remove prefix underscores.
  /// - add prefix if the leading character is a number.
  /// Returns empty string if not possible.
  static std::string ConvertToValidName(const std::string& name);
  static std::string NameTensor(const TensorMetadata& tensor,
                                const std::string& default_name);
  static void ResolveConflictedInputAndOutputNames(
      std::vector<std::string>* input, std::vector<std::string>* output);
};

}  // namespace codegen
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_CODE_GENERATOR_H_
