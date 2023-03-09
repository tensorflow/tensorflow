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

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/python/framework/op_reg_offset.pb.h"
#include "tensorflow/python/framework/python_op_gen.h"
#include "tensorflow/tsl/lib/io/buffered_inputstream.h"
#include "tensorflow/tsl/lib/io/random_inputstream.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/str_util.h"
#include "tensorflow/tsl/util/command_line_flags.h"

namespace tensorflow {
namespace {

constexpr char kUsage[] =
    "This tool generates python wrapper for tensorflow ops.";

Status ReadOpListFromFile(const string& filename,
                          std::vector<string>* op_list) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(filename, &file));
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
  return OkStatus();
}

Status ReadOpRegOffsetsFromFile(absl::string_view filename,
                                OpRegOffsets* op_reg_offsets) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(
      Env::Default()->NewRandomAccessFile(std::string(filename), &file));
  io::RandomAccessInputStream input_stream(file.get());
  io::BufferedInputStream in(&input_stream, 1 << 20);
  string contents;
  TF_RETURN_IF_ERROR(in.ReadAll(&contents));
  op_reg_offsets->ParseFromString(contents);
  return OkStatus();
}

std::vector<string> GetSourceFileListFromOpRegOffsets(
    const OpRegOffsets& offsets) {
  std::unordered_set<string> source_file_list;
  for (const auto& offset : offsets.offsets()) {
    source_file_list.insert(offset.filepath());
  }
  return std::vector<string>(source_file_list.begin(), source_file_list.end());
}

// Generates Python wapper functions for the registered ops given ApiDefs in
// `api_def_dirs` and write the result to `out_path` or print to stdout if
// `out_path` is empty.
//
// The ops in `hidden_op_list` will be private in python and the ops in
// `op_allowlist` will be skipped.
//
// If `source_file_name` is not empty, a comment block will be generated
// to show the source file name that the generated file is generated from.
Status PrintAllPythonOps(
    absl::Span<const string> api_def_dirs,
    absl::Span<const string> source_file_list, const string& out_path,
    const OpRegOffsets& op_reg_offsets,
    absl::Span<const string> op_allowlist = {},
    absl::Span<const string> hidden_op_list = {},
    const std::unordered_set<string>& type_annotate_ops = {}) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);
  if (!api_def_dirs.empty()) {
    Env* env = Env::Default();

    for (const auto& api_def_dir : api_def_dirs) {
      std::vector<string> api_files;
      TF_RETURN_IF_ERROR(env->GetMatchingPaths(
          io::JoinPath(api_def_dir, "*.pbtxt"), &api_files));
      TF_RETURN_IF_ERROR(api_def_map.LoadFileList(env, api_files));
    }
    api_def_map.UpdateDocs();
  }

  OpList pruned_ops;
  if (!op_allowlist.empty()) {
    std::unordered_set<string> allowlist(op_allowlist.begin(),
                                         op_allowlist.end());
    for (const auto& op_def : ops.op()) {
      if (allowlist.find(op_def.name()) != allowlist.end()) {
        *pruned_ops.mutable_op()->Add() = op_def;
      }
    }
  } else {
    pruned_ops = ops;
  }

  string result =
      GetPythonOps(pruned_ops, api_def_map, op_reg_offsets, hidden_op_list,
                   source_file_list, type_annotate_ops);

  if (out_path.empty()) {
    printf("%s", result.c_str());
  } else {
    std::unique_ptr<tensorflow::WritableFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewWritableFile(out_path, &file));
    TF_RETURN_IF_ERROR(file->Append(result));
  }

  return OkStatus();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  std::string api_def_dirs_raw;
  std::string op_allowlist_raw;
  std::string op_allowlist_filename;
  std::string hidden_op_list_raw;
  std::string hidden_op_list_filename;
  std::string op_reg_offset_filename;
  std::string out_path;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag(
          "api_def_dirs", &api_def_dirs_raw,
          "A comma separated directory list of where the api def files are."),
      tsl::Flag("op_allowlist", &op_allowlist_raw,
                "A comma separated list of allowed op names. All other ops "
                "will be ignored. op_allowlist and op_allowlist_filename "
                "cannot be set at the same time."),
      tsl::Flag("op_allowlist_filename", &op_allowlist_filename,
                "The name of the file that contains a list of allowed ops. "
                "op_allowlist and op_allowlist_filename cannot be set at the "
                "same time."),
      tsl::Flag("hidden_op_list", &hidden_op_list_raw,
                "A comma separated list of hidden op names. hidden_op_list and "
                "hidden_op_list_filename cannot be set at the same time."),
      tsl::Flag("hidden_op_list_filename", &hidden_op_list_filename,
                "The name of the file that contains a list of hidden ops. "
                "hidden_op_list and hidden_op_list_filename cannot be set at "
                "the same time."),
      tsl::Flag("op_reg_offset_filename", &op_reg_offset_filename,
                "The name of the file that contains mapping between op names "
                "and its location of op registration."),
      tsl::Flag("out_path", &out_path,
                "The destination of the output Python source. The result will "
                "be printed into stdout if out_path is empty."),
  };
  const std::string kUsageString = absl::StrCat(
      tensorflow::kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  const bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_result ||
      (!op_allowlist_raw.empty() && !op_allowlist_filename.empty()) ||
      (!hidden_op_list_raw.empty() && !hidden_op_list_filename.empty())) {
    LOG(ERROR) << kUsageString;
    return -1;
  }
  std::vector<std::string> op_allowlist;
  if (!op_allowlist_raw.empty()) {
    op_allowlist =
        tsl::str_util::Split(op_allowlist_raw, ',', tsl::str_util::SkipEmpty());
  } else if (!op_allowlist_filename.empty()) {
    TF_CHECK_OK(
        tensorflow::ReadOpListFromFile(op_allowlist_filename, &op_allowlist));
  }

  std::vector<std::string> hidden_op_list;
  if (!hidden_op_list_raw.empty()) {
    hidden_op_list = tsl::str_util::Split(hidden_op_list_raw, ',',
                                          tsl::str_util::SkipEmpty());
  } else if (!hidden_op_list_filename.empty()) {
    TF_CHECK_OK(tensorflow::ReadOpListFromFile(hidden_op_list_filename,
                                               &hidden_op_list));
  }

  tensorflow::OpRegOffsets op_reg_offsets;
  if (!op_reg_offset_filename.empty()) {
    TF_CHECK_OK(tensorflow::ReadOpRegOffsetsFromFile(op_reg_offset_filename,
                                                     &op_reg_offsets));
  }

  std::vector<std::string> source_file_list =
      tensorflow::GetSourceFileListFromOpRegOffsets(op_reg_offsets);

  std::vector<std::string> api_def_dirs =
      tsl::str_util::Split(api_def_dirs_raw, ",", tsl::str_util::SkipEmpty());

  // Add op name here to generate type annotations for it
  const std::unordered_set<std::string> type_annotate_ops{
      "FusedBatchNorm", "Add", "DynamicStitch"};

  TF_CHECK_OK(tensorflow::PrintAllPythonOps(
      api_def_dirs, source_file_list, out_path, op_reg_offsets, op_allowlist,
      hidden_op_list, type_annotate_ops));

  return 0;
}
