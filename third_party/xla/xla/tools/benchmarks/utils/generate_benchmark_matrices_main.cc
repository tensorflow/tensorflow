// Copyright 2025 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <iostream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "xla/tools/benchmarks/utils/generate_benchmark_matrices.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

int main(int argc, char* argv[]) {
  // Define the variable to hold the flag value BEFORE parsing
  std::string registry_file_path_arg;

  // Create the list of flags for tsl::Flags::Parse
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("registry_file",
                &registry_file_path_arg,  // Use the local variable
                "Path to the benchmark registry file (TextProto format).")
      // Add other tsl::Flags here if needed
  };

  // Parse flags using tsl::Flags::Parse
  // Important: Pass the original argc and argv. It modifies them.
  // Parse *before* InitMain, as InitMain might consume some flags.
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    // Note: tsl::Flags::Parse usually prints usage on error,
    // but adding a QFATAL provides explicit failure logging.
    LOG(QFATAL) << "Failed to parse command-line flags using tsl::Flags::Parse";
  }

  // Initialize main (potentially consumes other flags like --logtostderr)
  // Pass the potentially modified argc and argv from tsl::Flags::Parse
  tsl::port::InitMain(argv[0], &argc, &argv);

  // Check if the required flag was provided (now using the local variable)
  if (registry_file_path_arg.empty()) {
    // Construct usage string dynamically
    std::string usage =
        absl::StrCat("Usage: ", (argc > 0 ? argv[0] : "generate_matrix_main"),
                     " --registry_file=<path>");
    LOG(QFATAL) << "Required flag --registry_file is missing.\n" << usage;
  }

  // Resolve the path using the library function (use the parsed flag value)
  absl::StatusOr<std::string> resolved_registry_path =
      xla::tools::benchmarks::ResolveRegistryPath(registry_file_path_arg);

  // Use TF_QCHECK_OK for cleaner error checking with StatusOr
  TF_QCHECK_OK(resolved_registry_path.status())
      << "Failed to resolve registry file path";
  LOG(INFO) << "Using final registry path: " << *resolved_registry_path;

  // Parse the registry file
  absl::StatusOr<xla::tools::benchmarks::BenchmarkSuite> suite =
      xla::tools::benchmarks::ParseRegistry(*resolved_registry_path);

  TF_QCHECK_OK(suite.status()) << "Failed to parse benchmark registry file";

  // Generate the matrix
  Json::Value matrix_output = xla::tools::benchmarks::GenerateMatrix(*suite);

  // Output JSON matrix to stdout using Json::StreamWriterBuilder for better
  // control
  Json::StreamWriterBuilder writer_builder;
  writer_builder["indentation"] = "  ";  // 2-space indentation
  std::string output_string = Json::writeString(writer_builder, matrix_output);

  std::cout << output_string << std::endl;

  return 0;
}
