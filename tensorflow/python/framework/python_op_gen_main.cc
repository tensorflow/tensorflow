/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/framework/python_op_gen.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

Status ReadOpListFromFile(const string& filename,
                          std::vector<string>* op_list) {
  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(Env::Default()->NewRandomAccessFile(filename, &file));
  std::unique_ptr<io::InputBuffer> input_buffer(
      new io::InputBuffer(file.get(), 256 << 10));
  string line_contents;
  Status s = input_buffer->ReadLine(&line_contents);
  while (s.ok()) {
    // The parser assumes that the op name is the first string on each
    // line with no preceding whitespace, and ignores lines that do
    // not start with an op name as a comment.
    strings::Scanner scanner{StringPiece(line_contents)};
    StringPiece op_name;
    if (scanner.One(strings::Scanner::LETTER_DIGIT_DOT)
            .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
            .GetResult(nullptr, &op_name)) {
      op_list->emplace_back(op_name);
    }
    s = input_buffer->ReadLine(&line_contents);
  }
  if (!errors::IsOutOfRange(s)) return s;
  return Status::OK();
}

// The argument parsing is deliberately simplistic to support our only
// known use cases:
//
// 1. Read all op names from a file.
// 2. Read all op names from the arg as a comma-delimited list.
//
// Expected command-line argument syntax:
// ARG ::= '@' FILENAME
//       |  OP_NAME [',' OP_NAME]*
//       |  ''
Status ParseOpListCommandLine(const char* arg, std::vector<string>* op_list) {
  std::vector<string> op_names = str_util::Split(arg, ',');
  if (op_names.size() == 1 && op_names[0].empty()) {
    return Status::OK();
  } else if (op_names.size() == 1 && op_names[0].substr(0, 1) == "@") {
    const string filename = op_names[0].substr(1);
    return tensorflow::ReadOpListFromFile(filename, op_list);
  } else {
    *op_list = std::move(op_names);
  }
  return Status::OK();
}

// Use the name of the current executable to infer the C++ source file
// where the REGISTER_OP() call for the operator can be found.
// Returns the name of the file.
// Returns an empty string if the current executable's name does not
// follow a known pattern.
string InferSourceFileName(const char* argv_zero) {
  StringPiece command_str = io::Basename(argv_zero);

  // For built-in ops, the Bazel build creates a separate executable
  // with the name gen_<op type>_ops_py_wrappers_cc containing the
  // operators defined in <op type>_ops.cc
  const char* kExecPrefix = "gen_";
  const char* kExecSuffix = "_py_wrappers_cc";
  if (absl::ConsumePrefix(&command_str, kExecPrefix) &&
      str_util::EndsWith(command_str, kExecSuffix)) {
    command_str.remove_suffix(strlen(kExecSuffix));
    return strings::StrCat(command_str, ".cc");
  } else {
    return string("");
  }
}

void PrintAllPythonOps(const std::vector<string>& op_list,
                       const std::vector<string>& api_def_dirs,
                       const string& source_file_name,
                       bool op_list_is_allowlist,
                       const std::unordered_set<string> type_annotate_ops) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);
  if (!api_def_dirs.empty()) {
    Env* env = Env::Default();

    for (const auto& api_def_dir : api_def_dirs) {
      std::vector<string> api_files;
      TF_CHECK_OK(env->GetMatchingPaths(io::JoinPath(api_def_dir, "*.pbtxt"),
                                        &api_files));
      TF_CHECK_OK(api_def_map.LoadFileList(env, api_files));
    }
    api_def_map.UpdateDocs();
  }

  if (op_list_is_allowlist) {
    std::unordered_set<string> allowlist(op_list.begin(), op_list.end());
    OpList pruned_ops;
    for (const auto& op_def : ops.op()) {
      if (allowlist.find(op_def.name()) != allowlist.end()) {
        *pruned_ops.mutable_op()->Add() = op_def;
      }
    }
    PrintPythonOps(pruned_ops, api_def_map, {}, source_file_name,
                   type_annotate_ops);
  } else {
    PrintPythonOps(ops, api_def_map, op_list, source_file_name,
                   type_annotate_ops);
  }
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::string source_file_name =
      tensorflow::InferSourceFileName(argv[0]);

  // Usage:
  //   gen_main api_def_dir1,api_def_dir2,...
  //       [ @FILENAME | OpName[,OpName]* ] [0 | 1]
  if (argc < 2) {
    return -1;
  }
  std::vector<tensorflow::string> api_def_dirs = tensorflow::str_util::Split(
      argv[1], ",", tensorflow::str_util::SkipEmpty());

  // Add op name here to generate type annotations for it
  const std::unordered_set<tensorflow::string> type_annotate_ops{};

  if (argc == 2) {
    tensorflow::PrintAllPythonOps({}, api_def_dirs, source_file_name,
                                  false /* op_list_is_allowlist */,
                                  type_annotate_ops);
  } else if (argc == 3) {
    std::vector<tensorflow::string> hidden_ops;
    TF_CHECK_OK(tensorflow::ParseOpListCommandLine(argv[2], &hidden_ops));
    tensorflow::PrintAllPythonOps(hidden_ops, api_def_dirs, source_file_name,
                                  false /* op_list_is_allowlist */,
                                  type_annotate_ops);
  } else if (argc == 4) {
    std::vector<tensorflow::string> op_list;
    TF_CHECK_OK(tensorflow::ParseOpListCommandLine(argv[2], &op_list));
    tensorflow::PrintAllPythonOps(op_list, api_def_dirs, source_file_name,
                                  tensorflow::string(argv[3]) == "1",
                                  type_annotate_ops);
  } else {
    return -1;
  }
  return 0;
}
