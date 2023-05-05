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
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_TFLITE_SETTINGS_JSON_PARSER_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_TFLITE_SETTINGS_JSON_PARSER_H_

#include <string>

#include "flatbuffers/idl.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace delegates {
namespace utils {

// This class parses a JSON file to tflite::TFLiteSettings*.
// Note: This class is "thread-compatible", i.e. not thread-safe but also not
// thread-hostile
// <https://web.archive.org/web/20210125044505/https://www.ibm.com/developerworks/java/library/j-jtp09263/index.html>.
// That is, each instance is not thread-safe, but multiple separate instances
// are safely independent.
class TfLiteSettingsJsonParser {
 public:
  TfLiteSettingsJsonParser();

  // Loads TFLiteSettings from a JSON file path. The lifetime of the
  // TFLiteSettings object is tied to the lifetime of the
  // TfLiteSettingsJsonParser instance.
  //
  // Returns the pointer to the TFLiteSettings object or nullptr if an error is
  // encountered.
  const TFLiteSettings* Parse(const std::string& json_file_path);

  // Returns the buffer pointer to the loaded TFLiteSettings object or nullptr
  // if an error was encountered during loading or the TFLiteSettings object is
  // not loaded. The lifetime of the buffer is tied to the lifetime of the
  // TfLiteSettingsJsonParser instance.
  const uint8_t* GetBufferPointer();

  // Returns the buffer size of the loaded TFLiteSettings object or 0 if an
  // error was encountered during loading or the TFLiteSettings object is not
  // loaded.
  flatbuffers::uoffset_t GetBufferSize();

 private:
  // Parses content inside `json_file_path` into flatbuffer. Returns true if the
  // parsing was successful, otherwise the method returns false.
  bool LoadFromJsonFile(const std::string& json_file_path);

  flatbuffers::Parser parser_;
  uint8_t* buffer_pointer_;
  flatbuffers::uoffset_t buffer_size_;
};

}  // namespace utils
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_TFLITE_SETTINGS_JSON_PARSER_H_
