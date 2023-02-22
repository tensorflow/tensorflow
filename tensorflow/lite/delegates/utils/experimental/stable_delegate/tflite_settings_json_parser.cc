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
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"

#include <string>

#include "flatbuffers/idl.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_fbs_contents-inl.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace delegates {
namespace utils {

TfLiteSettingsJsonParser::TfLiteSettingsJsonParser() {
  TFLITE_DCHECK(parser_.Parse(configuration_fbs_contents) &&
                parser_.SetRootType("TFLiteSettings"));
}

const TFLiteSettings* TfLiteSettingsJsonParser::Parse(
    const std::string& json_file_path) {
  if (!LoadFromJsonFile(json_file_path) || buffer_pointer_ == nullptr) {
    return nullptr;
  }
  return flatbuffers::GetRoot<TFLiteSettings>(buffer_pointer_);
}

const uint8_t* TfLiteSettingsJsonParser::GetBufferPointer() {
  return buffer_pointer_;
}

flatbuffers::uoffset_t TfLiteSettingsJsonParser::GetBufferSize() {
  return buffer_size_;
}

bool TfLiteSettingsJsonParser::LoadFromJsonFile(
    const std::string& json_file_path) {
  buffer_size_ = 0;
  buffer_pointer_ = nullptr;
  if (json_file_path.empty()) {
    TFLITE_LOG(ERROR) << "Invalid JSON file path.";
    return false;
  }
  std::string json_file;
  if (!flatbuffers::LoadFile(json_file_path.c_str(), false, &json_file)) {
    TFLITE_LOG(ERROR) << "Failed to load the delegate settings file ("
                      << json_file_path << ").";
    return false;
  }
  if (!parser_.Parse(json_file.c_str())) {
    TFLITE_LOG(ERROR) << "Failed to parse the delegate settings file ("
                      << json_file_path << ").";
    return false;
  }
  buffer_size_ = parser_.builder_.GetSize();
  buffer_pointer_ = parser_.builder_.GetBufferPointer();
  return true;
}

}  // namespace utils
}  // namespace delegates
}  // namespace tflite
