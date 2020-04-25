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
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/strip.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

const char kInputModelFlag[] = "input_model";
const char kNamespace[] = "namespace";
const char kOutputRegistrationFlag[] = "output_registration";
const char kTfLitePathFlag[] = "tflite_path";
const char kForMicro[] = "for_micro";

void ParseFlagAndInit(int* argc, char** argv, string* input_model,
                      string* output_registration, string* tflite_path,
                      string* namespace_flag, bool* for_micro) {
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kInputModelFlag, input_model,
                               "path to the tflite model"),
      tflite::Flag::CreateFlag(kOutputRegistrationFlag, output_registration,
                               "filename for generated registration code"),
      tflite::Flag::CreateFlag(kTfLitePathFlag, tflite_path,
                               "Path to tensorflow lite dir"),
      tflite::Flag::CreateFlag(
          kNamespace, namespace_flag,
          "Namespace in which to put RegisterSelectedOps."),
      tflite::Flag::CreateFlag(
          kForMicro, for_micro,
          "By default this script generate TFL registration file, but can "
          "also generate TFLM files when this flag is set to true"),
  };

  tflite::Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
}

namespace {

void GenerateFileContent(const std::string& tflite_path,
                         const std::string& filename,
                         const std::string& namespace_flag,
                         const tflite::RegisteredOpMap& builtin_ops,
                         const tflite::RegisteredOpMap& custom_ops,
                         const bool for_micro) {
  std::ofstream fout(filename);

  if (for_micro) {
    if (!builtin_ops.empty()) {
      fout << "#include \"" << tflite_path << "/micro/kernels/micro_ops.h\"\n";
    }
    fout << "#include \"" << tflite_path
         << "/micro/micro_mutable_op_resolver.h\"\n";
  } else {
    if (!builtin_ops.empty()) {
      fout << "#include \"" << tflite_path
           << "/kernels/builtin_op_kernels.h\"\n";
    }
    fout << "#include \"" << tflite_path << "/model.h\"\n";
    fout << "#include \"" << tflite_path << "/op_resolver.h\"\n";
  }

  if (!custom_ops.empty()) {
    fout << "namespace tflite {\n";
    fout << "namespace ops {\n";
    fout << "namespace custom {\n";
    fout << "// Forward-declarations for the custom ops.\n";
    for (const auto& op : custom_ops) {
      fout << "TfLiteRegistration* Register_"
           << ::tflite::NormalizeCustomOpName(op.first) << "();\n";
    }
    fout << "}  // namespace custom\n";
    fout << "}  // namespace ops\n";
    fout << "}  // namespace tflite\n";
  }

  if (!namespace_flag.empty()) {
    fout << "namespace " << namespace_flag << " {\n";
  }
  if (for_micro) {
    fout << "void RegisterSelectedOps(::tflite::MicroMutableOpResolver* "
            "resolver) {\n";
  } else {
    fout << "void RegisterSelectedOps(::tflite::MutableOpResolver* resolver) "
            "{\n";
  }
  for (const auto& op : builtin_ops) {
    fout << "  resolver->AddBuiltin(::tflite::BuiltinOperator_" << op.first;
    if (for_micro) {
      fout << ", ::tflite::ops::micro::Register_" << op.first << "()";
    } else {
      fout << ", ::tflite::ops::builtin::Register_" << op.first << "()";
    }
    if (op.second.first != 1 || op.second.second != 1) {
      fout << ", " << op.second.first << ", " << op.second.second;
    }
    fout << ");\n";
  }
  for (const auto& op : custom_ops) {
    fout << "  resolver->AddCustom(\"" << op.first
         << "\", ::tflite::ops::custom::Register_"
         << ::tflite::NormalizeCustomOpName(op.first) << "()";
    if (op.second.first != 1 || op.second.second != 1) {
      fout << ", " << op.second.first << ", " << op.second.second;
    }
    fout << ");\n";
  }
  fout << "}\n";
  if (!namespace_flag.empty()) {
    fout << "}  // namespace " << namespace_flag << "\n";
  }
  fout.close();
}

void AddOpsFromModel(const string& input_model,
                     tflite::RegisteredOpMap* builtin_ops,
                     tflite::RegisteredOpMap* custom_ops) {
  std::ifstream fin(input_model);
  std::stringstream content;
  content << fin.rdbuf();
  // Need to store content data first, otherwise, it won't work in bazel.
  string content_str = content.str();
  const ::tflite::Model* model = ::tflite::GetModel(content_str.data());
  ::tflite::ReadOpsFromModel(model, builtin_ops, custom_ops);
}

}  // namespace

int main(int argc, char** argv) {
  string input_model;
  string output_registration;
  string tflite_path;
  string namespace_flag;
  bool for_micro = false;
  ParseFlagAndInit(&argc, argv, &input_model, &output_registration,
                   &tflite_path, &namespace_flag, &for_micro);

  tflite::RegisteredOpMap builtin_ops;
  tflite::RegisteredOpMap custom_ops;
  if (!input_model.empty()) {
    AddOpsFromModel(input_model, &builtin_ops, &custom_ops);
  }
  for (int i = 1; i < argc; i++) {
    AddOpsFromModel(argv[i], &builtin_ops, &custom_ops);
  }
  GenerateFileContent(tflite_path, output_registration, namespace_flag,
                      builtin_ops, custom_ops, for_micro);
  return 0;
}
