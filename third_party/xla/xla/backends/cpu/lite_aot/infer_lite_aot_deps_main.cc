/* Copyright 2026 The OpenXLA Authors.

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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/lite_aot/infer_lite_aot_deps_main_lib.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace {
struct InferLiteAotDepsOptions {
  std::string compilation_result_path;
  std::string output_deps_path;
};

std::vector<tsl::Flag> GetFlagList(InferLiteAotDepsOptions* options) {
  return {
      tsl::Flag("compilation_result", &options->compilation_result_path,
                "Path to the CompilationResult protobuf file."),
      tsl::Flag("output_deps", &options->output_deps_path,
                "Path to the output file to write dependencies. If not "
                "provided, prints to console."),
  };
}
}  // namespace

int main(int argc, char** argv) {
  InferLiteAotDepsOptions options;
  std::vector<tsl::Flag> flag_list = GetFlagList(&options);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 1;
  }

  if (options.compilation_result_path.empty()) {
    LOG(ERROR) << "Missing mandatory --compilation_result flag.\n" << usage;
    return 1;
  }

  xla::cpu::CompilationResultProto compilation_result;
  absl::Status read_status =
      tsl::ReadBinaryProto(tsl::Env::Default(), options.compilation_result_path,
                           &compilation_result);
  if (!read_status.ok()) {
    LOG(ERROR) << "Failed to read CompilationResult from "
               << options.compilation_result_path << ": " << read_status;
    return 1;
  }

  absl::StatusOr<std::vector<std::string>> deps =
      xla::cpu::InferLiteAotDeps(compilation_result);

  if (!deps.ok()) {
    LOG(ERROR) << "Failed to infer dependencies: " << deps.status();
    return 1;
  }

  std::string output = absl::StrJoin(*deps, "\n");
  if (!output.empty()) {
    output += '\n';
  }

  if (options.output_deps_path.empty()) {
    LOG(INFO) << "Dependencies:\n" << output;
  } else {
    absl::Status write_status = tsl::WriteStringToFile(
        tsl::Env::Default(), options.output_deps_path, output);
    if (!write_status.ok()) {
      LOG(ERROR) << "Failed to write dependencies to "
                 << options.output_deps_path << ": " << write_status;
      return 1;
    }
  }

  return 0;
}
