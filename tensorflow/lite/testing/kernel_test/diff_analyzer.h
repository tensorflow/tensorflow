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
#ifndef TENSORFLOW_LITE_TESTING_KERNEL_TEST_DIFF_ANALYZER_H_
#define TENSORFLOW_LITE_TESTING_KERNEL_TEST_DIFF_ANALYZER_H_

#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

// Reads the baseline and test files with output tensor values, and calculates
// the diff metrics.
class DiffAnalyzer {
 public:
  DiffAnalyzer() = default;
  TfLiteStatus ReadFiles(const string& base, const string& test);
  TfLiteStatus WriteReport(const string& filename);

 private:
  std::vector<std::vector<float>> base_tensors_;
  std::vector<std::vector<float>> test_tensors_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_KERNEL_TEST_DIFF_ANALYZER_H_
