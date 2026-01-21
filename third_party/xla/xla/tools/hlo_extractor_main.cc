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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/util.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace xla {

static constexpr absl::string_view kUsage = R"(
This tool lets you extract individual instructions from the HLO module in the
file, plus all the other instructions reachable through the parameters into a
separate HLO module. This is useful when you want to debug a single custom call
or a fusion from a larger module.

Usage:

  bazel run hlo-extractor -- --instruction=instr.name path/to/hlo_module
)";

struct HloExtractorConfig {
  std::string input_file;
  std::string input_format;
  std::string instruction;
  int64_t height = 0;
};

static std::string GetHloPath(const HloExtractorConfig& opts, int argc,
                              char** argv) {
  if (!opts.input_file.empty()) {
    return opts.input_file;
  }
  CHECK(argc == 2) << "Must specify a single input file";
  return argv[1];
}

static HloInstruction* FindInstruction(const HloModule* module,
                                       absl::string_view name) {
  for (const HloComputation* computation : module->computations()) {
    if (HloInstruction* instruction =
            hlo_query::FindInstruction(computation, name)) {
      return instruction;
    }
  }
  return nullptr;
}

static absl::Status RunHloExtractor(const HloExtractorConfig& opts, int argc,
                                    char** argv) {
  std::string hlo_path = GetHloPath(opts, argc, argv);

  std::string module_str;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &module_str));

  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(hlo_path));
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      LoadModuleFromData(module_str, format));

  HloInstruction* instruction = FindInstruction(module.get(), opts.instruction);
  if (!instruction) {
    return InvalidArgument(
        "Instruction '%s' was not found in the hlo module %s", opts.instruction,
        hlo_path);
  }

  std::unique_ptr<HloModule> extracted =
      ExtractModule(instruction, opts.height);
  std::cout << extracted->ToString();

  return absl::OkStatus();
}

}  // namespace xla

using namespace xla;  // NOLINT

int main(int argc, char** argv) {
  HloExtractorConfig opts;

  std::vector<tsl::Flag> flags = {
      tsl::Flag("format", &opts.input_format,
                "The format of the input file. By default inferred from the "
                "filename. Valid values:\n"
                "\t\t\t  txt   : HLO textual format\n"
                "\t\t\t  pb    : xla::HloProto in binary proto format"),

      tsl::Flag("instruction", &opts.instruction,
                "The name of the instruction to extract"),

      tsl::Flag(
          "height", &opts.height,
          "The number of instructions to follow from parameters before "
          "extracing a computation. The value of `-1` means that new "
          "computation will include all transitive operands of `instruction`. "
          "The value of `0` (default) will extract a single instruction."),
  };
  xla::AppendDebugOptionsFlags(&flags);

  // Try to parse `hlo-extractor` flags first.
  std::string usage =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flags));
  if (!tsl::Flags::Parse(&argc, argv, flags)) {
    std::cerr << usage;
    return 1;
  }

  // Initialize main process and maybe parse additional extra flags.
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (absl::Status status = RunHloExtractor(opts, argc, argv); !status.ok()) {
    std::cerr << status;
    return 1;
  }

  return 0;
}
