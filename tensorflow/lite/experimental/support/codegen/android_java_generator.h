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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_ANDROID_JAVA_GENERATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_ANDROID_JAVA_GENERATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/support/codegen/code_generator.h"
#include "tensorflow/lite/experimental/support/codegen/utils.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace support {
namespace codegen {

namespace details_android_java {

/// The intermediate data structure for generating code from TensorMetadata.
/// Should only be used as const reference when created.
struct TensorInfo {
  std::string name;
  std::string upper_camel_name;
  std::string content_type;
  std::string wrapper_type;
  std::string processor_type;
  bool is_input;
  /// Optional. Set to -1 if not applicable.
  int normalization_unit;
  /// Optional. Set to -1 if associated_axis_label is empty.
  int associated_axis_label_index;
  /// Optional. Set to -1 if associated_value_label is empty.
  int associated_value_label_index;
};

/// The intermediate data structure for generating code from ModelMetadata.
/// Should only be used as const reference when created.
struct ModelInfo {
  std::string package_name;
  std::string model_asset_path;
  std::string model_class_name;
  std::string model_versioned_name;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  // Extra helper fields. For models with inputs "a", "b" and outputs "x", "y":
  std::string input_type_param_list;
  // e.g. "TensorImage a, TensorBuffer b"
  std::string inputs_list;
  // e.g. "a, b"
  std::string postprocessor_type_param_list;
  // e.g. "ImageProcessor xPostprocessor, TensorProcessor yPostprocessor"
  std::string postprocessors_list;
  // e.g. "xPostprocessor, yPostprocessor"
};

}  // namespace details_android_java

constexpr char JAVA_EXT[] = ".java";

/// Generates Android supporting codes and modules (in Java) based on TFLite
/// metadata.
class AndroidJavaGenerator : public CodeGenerator {
 public:
  /// Creates an AndroidJavaGenerator.
  /// Args:
  /// - module_root: The root of destination Java module.
  explicit AndroidJavaGenerator(const std::string& module_root);

  /// Generates files. Returns the file paths and contents.
  /// Args:
  /// - model: The TFLite model with Metadata filled.
  /// - package_name: The name of the Java package which generated classes
  /// belong to.
  /// - model_class_name: A readable name of the generated wrapper class, such
  /// as "ImageClassifier", "MobileNetV2" or "MyModel".
  /// - model_asset_path: The relevant path to the model file in the asset.
  // TODO(b/141225157): Automatically generate model_class_name.
  GenerationResult Generate(const Model* model, const std::string& package_name,
                            const std::string& model_class_name,
                            const std::string& model_asset_path);

  /// Generates files and returns the file paths and contents.
  /// It's mostly identical with the previous one, but the model here is
  /// provided as binary flatbuffer content without parsing.
  GenerationResult Generate(const char* model_storage,
                            const std::string& package_name,
                            const std::string& model_class_name,
                            const std::string& model_asset_path);

  std::string GetErrorMessage();

 private:
  const std::string module_root_;
  ErrorReporter err_;
};

}  // namespace codegen
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_ANDROID_JAVA_GENERATOR_H_
