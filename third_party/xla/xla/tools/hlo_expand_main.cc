/* Copyright 2023 The OpenXLA Authors.

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
#include <iostream>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/tools/hlo_expand.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace {

const char* const kUsage = R"(
This tool lets you convert a HloModule from stdin or a file to another format,
run a set of expander passes, and dump the output to stdout or a file.

These are the supported formats:
1) a hlo text dump, the string should be in HloModule::ToString() format.
2) a binary or text proto file, the proto should be in xla.HloProto type.

Usage:

  hlo-expand \
    [--input_format=[hlo|pb|pbtxt]] \
    [--optional_flags] \
    [path/to/hlo_module]
)";

}  // namespace

// This expander tool is divided into the following steps:
// 1. Load HloModule from stdin or file.
// 2. Add a set of passes to the HloPassPipeline.
// 3. Run a set of passes on the module.
// 4a. Optionally print the output to stdout.
// 4b. Optionally write the output to file in the specified format.
int main(int argc, char** argv) {
  xla::HloExpandConfig config;
  auto flag_list = GetFlags(config);

  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  const std::string kHelpString = "Try: hlo-expand --help\n";

  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    std::cerr << absl::StrCat(
        "Error parsing command line flags. See usage below:\n", kUsageString,
        "\n");
    return 1;
  }

  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  if (config.help) {
    std::cerr << kUsageString;
    return 0;
  }

  if (argc > 2) {
    std::cerr << absl::StrCat(
        "Cannot parse more than one argument. See usage below:\n",
        kUsageString);
    return 1;
  }

  ParseCompoundFlags(config);

  std::string hlo_filename = argc == 2 ? argv[1] : "-";
  std::string output_filename =
      config.output_file.empty() ? "-" : config.output_file;

  // Validate input_format
  if (config.input_format.empty() && hlo_filename != "-") {
    config.input_format = std::string(tsl::io::Extension(hlo_filename));
  }
  if (config.input_format != "hlo" && config.input_format != "pb" &&
      config.input_format != "pbtxt" && config.input_format != "txt") {
    std::cerr << absl::StrCat(
        "input_format must be specified as [hlo|pb|pbtxt|txt].\n", kHelpString);
    return 1;
  }

  // If filename is not provided, use input_format flag to infer stdin type.
  if (hlo_filename == "-") {
    std::cout << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                 "interrupt)\n";
  }

  // Validate output_format
  if (config.output_format.empty()) {
    config.output_format =
        output_filename == "-"
            ? config.input_format
            : std::string(tsl::io::Extension(output_filename));
  }
  if (config.output_format != "hlo" && config.output_format != "pb" &&
      config.output_format != "pbtxt" && config.output_format != "txt") {
    std::cerr << absl::StrCat(
        "output_format must be specified as [hlo|pb|pbtxt].\n", kHelpString);
    return 1;
  }

  // 1. Load HloModule from stdin or file.
  absl::StatusOr<std::unique_ptr<xla::HloModule>> status_or_module;
  if (hlo_filename == "-") {
    std::string input;
    std::getline(std::cin, input, static_cast<char>(EOF));
    status_or_module = xla::LoadModuleFromData(input, config.input_format);
  } else {
    status_or_module =
        xla::LoadModuleFromFile(hlo_filename, config.input_format);
  }
  if (!status_or_module.ok()) {
    std::cerr << status_or_module.status() << "\nTry: hlo-expand --help\n";
    return 1;
  }

  // 2. Add a set of passes to the HloPassPipeline.
  xla::HloPassPipeline pipeline("expand_pass_pipeline");
  auto& hlo_module = status_or_module.value();
  AddPassesToPipeline(config, pipeline, hlo_module->config());

  // 3. Run a set of expander passes on the module.
  auto pipeline_status = pipeline.Run(hlo_module.get()).status();
  if (!pipeline_status.ok()) {
    std::cerr << pipeline_status;
    return 1;
  }

  // 4. Optionally print the output to stdout.
  if (output_filename == "-") {
    if (config.output_format == "hlo" || config.output_format == "txt") {
      std::cout << hlo_module->ToString();
    } else if (config.output_format == "pbtxt") {
      std::cout << hlo_module->ToProto().DebugString();
    } else {
      std::cerr << absl::StrCat(
          "Printing to stdout must specify supported "
          "output_format=[hlo|pbtxt|txt].\n",
          kHelpString);
      return 1;
    }
    return 0;
  }

  // 5. Optionally write the output to file in the specified format.
  absl::Status status;
  if (config.output_format == "hlo") {
    status = tsl::WriteStringToFile(tsl::Env::Default(), output_filename,
                                    hlo_module->ToString());
  } else if (config.output_format == "pb") {
    status = tsl::WriteBinaryProto(tsl::Env::Default(), output_filename,
                                   hlo_module->ToProto());
  } else if (config.output_format == "pbtxt") {
    status = tsl::WriteTextProto(tsl::Env::Default(), output_filename,
                                 hlo_module->ToProto());
  }

  if (!status.ok()) {
    std::cerr << status;
    return 1;
  }

  return 0;
}
