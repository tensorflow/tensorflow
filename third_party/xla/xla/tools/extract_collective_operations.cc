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

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo.pb.h"
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
This tool extracts collective operations (all-reduce and all-gather) from HLO module and saves them together
to the separate module.

Usage:
bazel run extract_collective_operations -- --input=path/to/hlo_module
  --output=path/to/hlo_module --operations=all-reduce,all-gather
)";
}  // namespace

namespace xla {

absl::Status ExtractCollectiveOperations(
    const std::string& input, const std::string& output,
    const absl::flat_hash_set<HloOpcode>& operation_types) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> test_module,
      LoadModuleFromFile(input, std::string(tsl::io::Extension(input)),
                         hlo_module_loader_details::Config(), nullptr));

  absl::flat_hash_set<HloOpcode> done_ops;
  absl::flat_hash_set<HloOpcode> non_optimized_ops;
  if (operation_types.contains(HloOpcode::kAllReduce)) {
    non_optimized_ops.insert(HloOpcode::kAllReduce);
    done_ops.insert(HloOpcode::kAllReduceDone);
  }
  if (operation_types.contains(HloOpcode::kAllGather)) {
    non_optimized_ops.insert(HloOpcode::kAllGather);
    done_ops.insert(HloOpcode::kAllGatherDone);
  }

  std::vector<xla::HloInstruction*> collective_instructions;
  for (const auto& op : test_module->computations()) {
    for (const auto& instr : op->instructions()) {
      if (operation_types.contains(HloOpcode::kAllReduce) &&
          HloPredicateIsOp<HloOpcode::kAllReduce, HloOpcode::kAllReduceStart,
                           HloOpcode::kAllReduceDone>(instr)) {
        collective_instructions.push_back(instr);
      }

      if (operation_types.contains(HloOpcode::kAllGather) &&
          HloPredicateIsOp<HloOpcode::kAllGather, HloOpcode::kAllGatherStart,
                           HloOpcode::kAllGatherDone>(instr)) {
        collective_instructions.push_back(instr);
      }
    }
  }

  if (collective_instructions.empty()) {
    return absl::InternalError("No collective instructions found.");
  }
  auto collectives_module = ExtractCollectiveOperationsIntoNewModule(
      collective_instructions, done_ops, non_optimized_ops);

  QCHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), output,
                                   collectives_module->ToString()))
      << "Can't open or write output module at " << output;
  return absl::OkStatus();
}
}  // namespace xla

int main(int argc, char** argv) {
  std::string input;
  std::string output;
  std::string operations;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input, "input file"),
      tsl::Flag("output", &output, "output file"),
      tsl::Flag("operations", &operations,
                "operations. possible values: all-reduce, all-gather")};
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  absl::flat_hash_set<xla::HloOpcode> operation_types;
  if (absl::StrContains(operations, "all-reduce")) {
    operation_types.insert(xla::HloOpcode::kAllReduce);
  }
  if (absl::StrContains(operations, "all-gather")) {
    operation_types.insert(xla::HloOpcode::kAllGather);
  }
  TF_CHECK_OK(xla::ExtractCollectiveOperations(input, output, operation_types));
  return 0;
}
