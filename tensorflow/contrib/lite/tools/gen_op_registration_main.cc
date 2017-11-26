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

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Flags;
using tensorflow::string;

namespace {

void GenerateFileContent(const string& filename,
                         const std::vector<string>& builtin_ops,
                         const std::vector<string>& custom_ops) {
  std::ofstream fout(filename);

  fout << "#include "
          "\"third_party/tensorflow/contrib/lite/model.h\"\n";
  fout << "#include "
          "\"third_party/tensorflow/contrib/lite/tools/mutable_op_resolver.h\"\n";
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
  std::vector<tensorflow::Flag> flag_list = {
      Flag("input_model", &input_model, "path to the tflite model"),
      Flag("output_registration", &output_registration,
           "filename for generated registration code"),
  };
  Flags::Parse(&argc, argv, flag_list);

  tensorflow::port::InitMain(argv[0], &argc, &argv);
  std::vector<string> builtin_ops;
  std::vector<string> custom_ops;

  std::ifstream fin(input_model);
  std::stringstream content;
  content << fin.rdbuf();
  const ::tflite::Model* model = ::tflite::GetModel(content.str().data());
  ::tflite::ReadOpsFromModel(model, &builtin_ops, &custom_ops);
  GenerateFileContent(output_registration, builtin_ops, custom_ops);
  return 0;
}
