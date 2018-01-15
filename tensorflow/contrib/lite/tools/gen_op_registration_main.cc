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

#include <cassert>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/strip.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

const char kInputModelFlag[] = "input_model";
const char kOutputRegistrationFlag[] = "output_registration";
const char kTfLitePathFlag[] = "tflite_path";

using tensorflow::Flag;
using tensorflow::Flags;
using tensorflow::string;

void ParseFlagAndInit(int argc, char** argv, string* input_model,
                      string* output_registration, string* tflite_path) {
  std::vector<tensorflow::Flag> flag_list = {
      Flag(kInputModelFlag, input_model, "path to the tflite model"),
      Flag(kOutputRegistrationFlag, output_registration,
           "filename for generated registration code"),
      Flag(kTfLitePathFlag, tflite_path, "Path to tensorflow lite dir"),
  };

  Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
}

namespace {

void GenerateFileContent(const std::string& tflite_path,
                         const std::string& filename,
                         const std::vector<string>& builtin_ops,
                         const std::vector<string>& custom_ops) {
  std::ofstream fout(filename);

  fout << "#include \"" << tflite_path << "/model.h\"\n";
  fout << "#include \"" << tflite_path << "/tools/mutable_op_resolver.h\"\n";

  fout << "namespace tflite {\n";
  fout << "namespace ops {\n";
  if (!builtin_ops.empty()) {
    fout << "namespace builtin {\n";
    fout << "// Forward-declarations for the builtin ops.\n";
    for (const auto& op : builtin_ops) {
      fout << "TfLiteRegistration* Register_" << op << "();\n";
    }
    fout << "}  // namespace builtin\n";
  }

  if (!custom_ops.empty()) {
    fout << "namespace custom {\n";
    fout << "// Forward-declarations for the custom ops.\n";
    for (const auto& op : custom_ops) {
      fout << "TfLiteRegistration* Register_"
           << ::tflite::NormalizeCustomOpName(op) << "();\n";
    }
    fout << "}  // namespace custom\n";
  }
  fout << "}  // namespace ops\n";
  fout << "}  // namespace tflite\n";

  fout << "void RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {\n";
  for (const auto& op : builtin_ops) {
    fout << "  resolver->AddBuiltin(::tflite::BuiltinOperator_" << op
         << ", ::tflite::ops::builtin::Register_" << op << "());\n";
  }
  for (const auto& op : custom_ops) {
    fout << "  resolver->AddCustom(\"" << op
         << "\", ::tflite::ops::custom::Register_"
         << ::tflite::NormalizeCustomOpName(op) << "());\n";
  }
  fout << "}\n";
  fout.close();
}
}  // namespace

int main(int argc, char** argv) {
  string input_model;
  string output_registration;
  string tflite_path;
  ParseFlagAndInit(argc, argv, &input_model, &output_registration,
                   &tflite_path);

  std::vector<string> builtin_ops;
  std::vector<string> custom_ops;
  std::ifstream fin(input_model);
  std::stringstream content;
  content << fin.rdbuf();
  // Need to store content data first, otherwise, it won't work in bazel.
  string content_str = content.str();
  const ::tflite::Model* model = ::tflite::GetModel(content_str.data());
  ::tflite::ReadOpsFromModel(model, &builtin_ops, &custom_ops);
  GenerateFileContent(tflite_path, output_registration, builtin_ops,
                      custom_ops);
  return 0;
}
