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

#include "tensorflow/python/eager/python_eager_op_gen.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/scanner.h"
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
      op_list->emplace_back(op_name.ToString());
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

void PrintAllPythonOps(const std::vector<string>& op_list, bool require_shapes,
                       bool op_list_is_whitelist) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  if (op_list_is_whitelist) {
    std::unordered_set<string> whitelist(op_list.begin(), op_list.end());
    OpList pruned_ops;
    for (const auto& op_def : ops.op()) {
      if (whitelist.find(op_def.name()) != whitelist.end()) {
        *pruned_ops.mutable_op()->Add() = op_def;
      }
    }
    PrintEagerPythonOps(pruned_ops, {}, require_shapes);
  } else {
    PrintEagerPythonOps(ops, op_list, require_shapes);
  }
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // Usage:
  //   gen_main [ @FILENAME | OpName[,OpName]* ] (0 | 1) [0 | 1]
  if (argc == 2) {
    tensorflow::PrintAllPythonOps({}, {}, tensorflow::string(argv[1]) == "1");
  } else if (argc == 3) {
    std::vector<tensorflow::string> hidden_ops;
    TF_CHECK_OK(tensorflow::ParseOpListCommandLine(argv[1], &hidden_ops));
    tensorflow::PrintAllPythonOps(hidden_ops,
                                  tensorflow::string(argv[2]) == "1",
                                  false /* op_list_is_whitelist */);
  } else if (argc == 4) {
    std::vector<tensorflow::string> op_list;
    TF_CHECK_OK(tensorflow::ParseOpListCommandLine(argv[1], &op_list));
    tensorflow::PrintAllPythonOps(op_list, tensorflow::string(argv[2]) == "1",
                                  tensorflow::string(argv[3]) == "1");
  } else {
    return -1;
  }
  return 0;
}
