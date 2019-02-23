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

#include <vector>

#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/testing/kernel_test/diff_analyzer.h"

int main(int argc, char** argv) {
  string base, test, output;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("base", &base, "Path to the base serialized tensor."),
      tensorflow::Flag("test", &test, "Path to the test serialized tensor."),
      tensorflow::Flag("output", &output, "Path to the output file."),
  };
  tensorflow::Flags::Parse(&argc, argv, flag_list);

  tflite::testing::DiffAnalyzer diff_analyzer;
  diff_analyzer.ReadFiles(base, test);
  diff_analyzer.WriteReport(output);
  return 0;
}
