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

// A tool for reading a HloModule from a HloProto file and execute the module on
// given platform(s). See kUsage for details.

#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_opt/opt_lib.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace {
const char* const kUsage = R"(
This tool lets you run a given HloModule from a file (or stdin) and convert it
to expanded HLO, fully optimized HLO, or a binary depending on options.

HLO passes are always run, unless the HLO module is already scheduled (has
is_scheduled=True).

You can also pass in debug option flags for the HloModule.

Usage:

  bazel run opt -- --platform=[gpu|cpu|...] path/to/hlo_module
)";

struct HloOptConfig {
  // Optional flags.
  bool help{false};
  bool split_input_file{false};
  std::string platform{"transforms"};
  std::string input_file{""};
  std::string input_format{""};
  std::string output_file{"-"};
  std::string stage{"hlo"};
  bool list_stages{false};
  std::string passes{""};
  bool list_passes{false};
  bool emit_proto{false};
};

}  // namespace

namespace xla {

namespace {

// Convention separator as set by mlir-opt tool.
const char* kOptSeparator = "// -----";

std::string GetHloPath(const HloOptConfig& opts, int argc, char** argv) {
  if (!opts.input_file.empty()) {
    return opts.input_file;
  }
  QCHECK(argc == 2) << "Must specify a single input file";
  return argv[1];
}

absl::StatusOr<std::string> GetHloContents(const HloOptConfig& opts, int argc,
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

absl::StatusOr<std::vector<std::unique_ptr<HloModule>>> GetModules(
    const HloOptConfig& opts, int argc, char** argv) {
  TF_ASSIGN_OR_RETURN(std::string module_data,
                      GetHloContents(opts, argc, argv));

  std::vector<std::string> hlos;
  if (opts.split_input_file) {
    hlos = absl::StrSplit(module_data, kOptSeparator);
  } else {
    hlos.push_back(module_data);
  }

  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(GetHloPath(opts, argc, argv)));
  }

  std::vector<std::unique_ptr<HloModule>> out;
  out.reserve(hlos.size());

  for (const std::string& hlo : hlos) {
    if (absl::StrContains(hlo, "// ---")) {
      if (opts.split_input_file) {
        return absl::InternalError(
            "Unexpected separator found, expected exactly '// -----', found "
            "'// ---'");
      } else {
        return absl::InternalError(
            "'// ---' separator found in input, but -split-input-file not "
            "specified");
      }
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        LoadModuleFromData(hlo, format));
    out.push_back(std::move(module));
  }
  return out;
}

std::unique_ptr<HloModule> GetDummyModule() {
  std::string hlo_text = R"(
      HloModule m
        ENTRY main {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
        ROOT res = f32[] multiply(a, b)
      })";
  return std::make_unique<HloModule>(hlo_text, HloModuleConfig());
}

absl::StatusOr<std::string> TranslateToStage(int argc, char** argv,
                                             const HloOptConfig& opts) {
  TF_ASSIGN_OR_RETURN(OptProvider * provider,
                      OptProvider::GetProviderForPlatform(opts.platform));

  if (opts.list_stages) {
    return absl::StrJoin(provider->SupportedStages(), "\n");
  }
  // Use a dummy module for "list-passes" because pipelines compilation
  // requires a module.
  if (opts.list_passes) {
    auto dummy_module = GetDummyModule();
    provider->RegisterProviderPasses(*dummy_module);
    return provider->GetRegisteredPassNames();
  }

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<HloModule>> modules,
                      GetModules(opts, argc, argv));
  if (opts.emit_proto) {
    std::string proto_str_combined;
    for (const auto& module : modules) {
      absl::StrAppend(&proto_str_combined,
                      xla::MakeHloProto(*module).DebugString(), "\n");
    }
    return proto_str_combined;
  }

  // Registration can be done using HloModuleConfig, but some
  // GPU pipelines APIs expects HloModule.
  // Assumption: All input modules have same HloModuleConfig.
  provider->RegisterProviderPasses(*modules[0].get());

  std::string out_combined;

  for (std::unique_ptr<HloModule>& m : modules) {
    std::optional<std::string> out;
    if (!opts.passes.empty()) {
      TF_ASSIGN_OR_RETURN(out, provider->BuildAndRunTransformPipeline(
                                   std::move(m), opts.passes));
    } else {
      TF_ASSIGN_OR_RETURN(out,
                          provider->GenerateStage(std::move(m), opts.stage));
    }
    if (!out.has_value()) {
      return absl::UnimplementedError("Stage not supported");
    }
    absl::StrAppend(&out_combined, *out, "\n");
  }

  return out_combined;
}

absl::Status RunOpt(int argc, char** argv, const HloOptConfig& opts) {
  TF_ASSIGN_OR_RETURN(std::string output, TranslateToStage(argc, argv, opts));
  if (opts.output_file == "-") {
    std::cout << output << std::endl;
  } else {
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), opts.output_file, output));
  }
  return absl::OkStatus();
}

// This function is parsing only the debug options file, because we cannot wait
// till all the flags are parsed. If the debug_options file exists, then we have
// to first consider the debug_options from that file, then XLA_FLAGS, and then
// the command line flags. Hence, we parse the debug_options file first.
std::optional<absl::string_view> GetDebugOptionsFileName(int argc,
                                                         char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    absl::string_view arg = argv[i];
    if (absl::StrContains(arg, "--debug_options_file")) {
      auto eq_idx = arg.find('=');
      if (eq_idx != absl::string_view::npos) {
        return arg.substr(eq_idx + 1);
      } else {
        LOG(QFATAL) << "No value provided for --debug_options_file. Expected "
                    << "--debug_options_file=<filename>";
      }
    }
  }
  return std::nullopt;
}

}  // namespace
}  // namespace xla

// All XLA compiler flags are supported.
// Use `--xla_gpu_target_config_filename` to specify the target config.
int main(int argc, char** argv) {
  HloOptConfig opts;
  std::string unused_debug_options_filename;
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
                "\t\t\t * buffer-assignment: Buffer Assignment\n"
                "\t\t\t * hlo-backend: HLO after backend passes\n"
                "\t\t\t * html: HTML dump\n"),
      tsl::Flag("list-stages", &opts.list_stages,
                "Print all supported stages for a given platform and exit"),
      tsl::Flag("split-input-file", &opts.split_input_file,
                "Splits the input file in pieces based on '// -----' "
                "substring, and processes each chunk independently"),
      tsl::Flag("passes", &opts.passes,
                "Comma-separated list of passes to run."),
      tsl::Flag("list-passes", &opts.list_passes,
                "Print all supported passes for a given platform and exit"),
      tsl::Flag("emit-proto", &opts.emit_proto,
                "Emit HLO in `textproto` format, "
                "no optimization passes are applied."),
      // This flag is parsed separately, not as part of the HloOptConfig
      // options. `unused_debug_options_filename` is introduced for a
      // documentation, not used by the tool.
      tsl::Flag("debug_options_file", &unused_debug_options_filename,
                "A file containing debug options to be passed to the HLO "
                "module. The file should contain a serialized DebugOptions "
                "proto message. The order of precedence: command line flags > "
                "XLA_FLAGS > debug_options_file > default flags.")};

  // Modifies global DebugOptions, populates flags with every flag available
  // from xla.proto.
  xla::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

  std::optional<absl::string_view> debugOptionsFilename =
      xla::GetDebugOptionsFileName(argc, argv);
  if (debugOptionsFilename.has_value()) {
    xla::ParseFlagsFromDebugOptionsFile(debugOptionsFilename.value());
  }
  xla::ParseDebugOptionFlagsFromEnv(true);

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  absl::Status s = xla::RunOpt(argc, argv, opts);
  if (!s.ok()) {
    std::cerr << s;
    return 1;
  }
  return 0;
}
