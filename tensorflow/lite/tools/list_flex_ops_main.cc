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
#include <fstream>
#include <iostream>
#include <sstream>

#include "absl/strings/str_split.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/list_flex_ops.h"

const char kInputModelsFlag[] = "graphs";

int main(int argc, char** argv) {
  std::string input_models;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kInputModelsFlag, &input_models,
                               "path to the tflite models, separated by comma.",
                               tflite::Flag::kRequired),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  std::vector<std::string> models = absl::StrSplit(input_models, ',');
  tflite::flex::OpKernelSet flex_ops;
  for (const std::string& model_file : models) {
    std::ifstream fin;
    fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fin.open(model_file);
    std::stringstream content;
    content << fin.rdbuf();

    // Need to store content data first, otherwise, it won't work in bazel.
    std::string content_str = content.str();
    const ::tflite::Model* model = ::tflite::GetModel(content_str.data());
    tflite::flex::AddFlexOpsFromModel(model, &flex_ops);
  }
  std::cout << tflite::flex::OpListToJSONString(flex_ops);
  return 0;
}
