/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_MODELS_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_MODELS_TEST_UTILS_H_

#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace tflite {
namespace models {
using Frames = std::vector<std::vector<float>>;
}  // namespace models
}  // namespace tflite

#ifndef __ANDROID__
#include "file/base/path.h"
#include "tensorflow/core/platform/test.h"

inline string TestDataPath() {
  return string(file::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                               "contrib/lite/models/testdata/"));
}
inline int TestInputSize(const tflite::models::Frames& input_frames) {
  return input_frames.size();
}
#else
inline string TestDataPath() {
  return string("third_party/tensorflow/contrib/lite/models/testdata/");
}

inline int TestInputSize(const tflite::models::Frames& input_frames) {
  // Android TAP is very slow, we only test the first 20 frames.
  return 20;
}
#endif

namespace tflite {
namespace models {

// Read float data from a comma-separated file:
// Each line will be read into a float vector.
// The return result will be a vector of float vectors.
void ReadFrames(const string& csv_file_path, Frames* frames) {
  std::ifstream csv_file(csv_file_path);
  string line;
  while (std::getline(csv_file, line, '\n')) {
    std::vector<float> fields;
    // Used by strtok_r internaly for successive calls on the same string.
    char* save_ptr = nullptr;

    // Tokenize the line.
    char* next_token =
        strtok_r(const_cast<char*>(line.c_str()), ",", &save_ptr);
    while (next_token != nullptr) {
      float f = strtod(next_token, nullptr);
      fields.push_back(f);
      next_token = strtok_r(nullptr, ",", &save_ptr);
    }
    frames->push_back(fields);
  }
  csv_file.close();
}

}  // namespace models
}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_MODELS_TEST_UTILS_H_
