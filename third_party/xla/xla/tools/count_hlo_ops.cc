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

// A tool for counting HLO ops in a list of HLO modules.
// See kUsage for details.

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/status/status_macros.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"

namespace {
const char* const kUsage = R"(
Counts the number of times each HLO opcode appears in a list of HLO modules.

Usage:
  bazel run count_hlo_ops -- --input=${comma_separated_list_of_hlo_modules} \
    --format=[hlo|pb|pbtxt]

Example commmand with HLO dumps from `--xla_dump_to=/tmp/hlo`:
  INPUTS=$(echo /tmp/hlo/*before*optimizations* | sed 's/ /,/g')
  bazel run count_hlo_ops -- --input=${INPUTS} --format=hlo

Example output:
  reshape                        4081
  broadcast                      2652
  multiply                       2090
  add                            1866
  parameter                      1721
  ...
)";
}  // namespace

namespace xla {
namespace {

absl::Status CountOps(const std::string& input, const std::string& format,
                      absl::flat_hash_map<HloOpcode, int>* counts) {
  ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                   LoadModuleFromFile(input, format, {}));
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* hlo : computation->instructions()) {
      HloOpcode opcode = hlo->opcode();
      if (!counts->contains(opcode)) {
        (*counts)[opcode] = 1;
      } else {
        (*counts)[opcode]++;
      }
    }
  }
  return absl::OkStatus();
}

void PrintCounts(const absl::flat_hash_map<xla::HloOpcode, int>& counts) {
  std::vector<std::pair<xla::HloOpcode, int>> sorted_counts;
  sorted_counts.reserve(counts.size());

  // NOLINTNEXTLINE: The iteration order doesn't need to be deterministic here.
  for (const auto& [opcode, count] : counts) {
    sorted_counts.push_back({opcode, count});
  }
  std::sort(sorted_counts.begin(), sorted_counts.end(),
            [](const std::pair<xla::HloOpcode, int>& a,
               const std::pair<xla::HloOpcode, int>& b) {
              return a.second > b.second;
            });

  for (const auto& [opcode, count] : sorted_counts) {
    std::cout << absl::StrFormat("%-30s ", xla::HloOpcodeString(opcode))
              << count << std::endl;
  }
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::string inputs, format;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &inputs, "A comma-separated list of input files."),
      tsl::Flag("format", &format, "hlo|pb|pbtxt"),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    // Print the usage using cerr to avoid truncation by LOG.
    std::cerr << kUsageString;
    return 1;
  }

  absl::flat_hash_map<xla::HloOpcode, int> counts;
  std::vector<std::string> input_list =
      absl::StrSplit(inputs, ',', absl::SkipEmpty());

  for (std::string& input : input_list) {
    input = absl::StripAsciiWhitespace(input);
    absl::Status status = xla::CountOps(input, format, &counts);
    if (!status.ok()) {
      std::cerr << "Error processing " << input << ": " << status << std::endl;
    }
  }
  xla::PrintCounts(counts);
  return 0;
}
