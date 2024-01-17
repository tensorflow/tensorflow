/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// A tool for reading a HloModule from a HloProto file and execute the module on
// given platform(s). See kUsage for details.

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tools/hlo_opt/opt_lib.h"
#include "xla/tools/run_hlo_module.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/util/command_line_flags.h"

namespace {
const char* const kUsage = R"(
This tool lets you run a given HloModule from a file (or stdin) and convert it
to expanded HLO, fully optimized HLO, or a binary depending on options.

You can also pass in debug option flags for the HloModule.

Usage:

  bazel run opt -- --platform=[gpu|cpu|...] path/to/hlo_module
)";

struct HloOptConfig {
  // Optional flags.
  bool help{false};
  bool split_input_file{false};
  std::string platform{"gpu"};
  std::string input_file{""};
  std::string input_format{""};
  std::string output_file{"-"};
  std::string stage{"hlo"};
  bool list_stages{false};
};

}  // namespace

namespace xla {

namespace {

std::string GetHloPath(const HloOptConfig& opts, int argc, char** argv) {
  if (!opts.input_file.empty()) {
    return opts.input_file;
  }
  QCHECK(argc == 2) << "Must specify a single input file";
  return argv[1];
}

StatusOr<std::string> GetHloContents(const HloOptConfig& opts, int argc,
                                     char** argv) {
  std::string hlo_path = GetHloPath(opts, argc, argv);
  if (hlo_path == "-") {
    std::string input;
    std::getline(std::cin, input, static_cast<char>(EOF));
    return input;
  }

  std::string data;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &data));
  return data;
}

StatusOr<std::unique_ptr<HloModule>> GetModule(const HloOptConfig& opts,
                                               int argc, char** argv) {
  TF_ASSIGN_OR_RETURN(std::string module_data,
                      GetHloContents(opts, argc, argv));

  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(GetHloPath(opts, argc, argv)));
  }
  return LoadModuleFromData(module_data, format);
}

StatusOr<std::string> TranslateToStage(int argc, char** argv,
                                       const HloOptConfig& opts) {
  TF_ASSIGN_OR_RETURN(OptProvider * provider,
                      OptProvider::ProviderForPlatform(opts.platform));

  if (opts.list_stages) {
    return absl::StrJoin(provider->SupportedStages(), "\n");
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      GetModule(opts, argc, argv));

  TF_ASSIGN_OR_RETURN(std::optional<std::string> out,
                      provider->GenerateStage(std::move(module), opts.stage));

  if (!out.has_value()) {
    return absl::UnimplementedError("Stage not supported");
  }

  return *out;
}

Status RunOpt(int argc, char** argv, const HloOptConfig& opts) {
  TF_ASSIGN_OR_RETURN(std::string output, TranslateToStage(argc, argv, opts));
  if (opts.output_file == "-") {
    std::cout << output << std::endl;
  } else {
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), opts.output_file, output));
  }
  return OkStatus();
}

}  // namespace
}  // namespace xla

// gpu_device_config_filename: Probably deserves it's own flag? Since in here it
// will affect more top-level logic?
int main(int argc, char** argv) {
  HloOptConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("o", &opts.output_file,
                "Output filename, or '-' for stdout (default)."),
      tsl::Flag("platform", &opts.platform,
                "The platform for which we perform the translation"),
      tsl::Flag("format", &opts.input_format,
                "The format of the input file. By default inferred from the "
                "filename. Valid values:\n"
                "\t\t\t  hlo : HLO textual format\n"
                "\t\t\t  pb : xla::HloProto in binary proto format\n"
                "\t\t\t  pbtxt : xla::HloProto in text proto format"),
      tsl::Flag("stage", &opts.stage,
                "Output stage to dump. "
                "Valid values depend on the platform, for GPUs:\n"
                "\t\t\t * hlo : HLO after all optimizations\n"
                "\t\t\t * llvm : LLVM IR\n"
                "\t\t\t * ptx : PTX dump\n"
                "\t\t\t * buffer-assignment: Buffer Assignment\n"),
      tsl::Flag("list-stages", &opts.list_stages,
                "Print all supported stages for a given platform and exit")};
  // Modifies global DebugOptions, populates flags with every flag available
  // from xla.proto.
  xla::AppendDebugOptionsFlags(&flag_list);
  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  xla::Status s = xla::RunOpt(argc, argv, opts);
  if (!s.ok()) {
    std::cerr << s;
    return 1;
  }
  return 0;
}
