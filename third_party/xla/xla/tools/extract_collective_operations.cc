/* Copyright 2024 The OpenXLA Authors.

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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo.pb.h"
#include "xla/status.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace {
const char* const kUsage = R"(
This tool extracts collective operations from HLO module and saves them together
to the separate module.

Usage:
bazel run extract_collective_operations -- --input=path/to/hlo_module
  --output=path/to/hlo_module
)";
}  // namespace

namespace xla {
Status ExtractCollectiveOperations(const std::string& input,
                                   const std::string& output) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> test_module,
      LoadModuleFromFile(input, std::string(tsl::io::Extension(input)),
                         hlo_module_loader_details::Config(), nullptr));

  std::vector<xla::HloInstruction*> collective_instructions;
  for (const auto& op : test_module->computations()) {
    for (const auto& instr : op->instructions()) {
      if (absl::StartsWith(instr->name(), "all-")) {
        collective_instructions.push_back(instr);
      }
    }
  }

  if (collective_instructions.empty()) {
    return absl::InternalError("No collective instructions found.");
  }
  auto collectives_module =
      ExtractInstructionIntoNewModule(collective_instructions);

  QCHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), output,
                                   collectives_module->ToString()))
      << "Can't open or write output module at " << output;
  return absl::OkStatus();
}
}  // namespace xla

int main(int argc, char** argv) {
  std::string input;
  std::string output;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input, "input file"),
      tsl::Flag("output", &output, "output file")};
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }
  TF_CHECK_OK(xla::ExtractCollectiveOperations(input, output));
  return 0;
}
