/* Copyright 2025 The OpenXLA Authors.

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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

// At the moment we only print output indexing maps but feel free to extend it.
const char* const kUsage = R"(
Prints the indexing maps for the output operands of the root instruction of the given HLO module. For example:
print_indexing <file.hlo> [--operand_id=0] [--output_id=0])";

namespace xla {

absl::Status Run(const std::string& filename, int operand_id, int output_id) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      LoadModuleFromFile(filename));
  auto root = module->entry_computation()->root_instruction();
  bool print_all = operand_id < 0;
  int get_operand_id = operand_id;
  if (print_all) {
    get_operand_id = 0;
  }
  mlir::MLIRContext ctx;
  VLOG(1) << "module:\n" << module->ToString() << std::endl;
  LOG(INFO) << "root instruction is: " << root->ToString() << std::endl;
  VLOG(1) << "root is tuple: " << root->shape().IsTuple();
  std::vector<int> output_ids;
  if (output_id < 0) {
    // Enumerate all outputs.
    if (root->shape().IsTuple()) {
      for (int i = 0; i < root->shape().tuple_shapes().size(); ++i) {
        output_ids.push_back(i);
      }
    } else {
      output_ids.push_back(0);
    }
  } else {
    output_ids.push_back(output_id);
  }

  // With vector of output_ids we can support multiple outputs (for tuples) but
  // they are not yet supported in ComputeOutputToInputIndexing. So for now we
  // short-circuit it to the only non-tuple output.

  for (int out_id : output_ids) {
    HloInstructionIndexing indexing =
        ComputeOutputToInputIndexing(root, out_id, &ctx);
    LOG(INFO) << absl::StrFormat("output id %d has %d indexing maps", out_id,
                                 indexing.indexing_maps.size());
    if (indexing.indexing_maps.empty()) {
      std::cout << "No indexing maps found for output " << out_id << std::endl;
      continue;
    }
    for (const auto& [operand_id, operand_maps] :
         llvm::enumerate(indexing.indexing_maps)) {
      LOG(INFO) << absl::StrFormat("operand id %d has %d indexing maps",
                                   operand_id, operand_maps.size());
      if (operand_id != get_operand_id && !print_all) {
        continue;
      }
      if (operand_maps.empty()) {
        continue;
      }
      std::cout << absl::StrFormat("Output %d operand %d:\n", out_id,
                                   operand_id);
      for (const auto& indexing_map : operand_maps) {
        std::cout << indexing_map << std::endl;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xla

int main(int argc, char** argv) {
  int operand_id = -1;
  int output_id = -1;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag(
          "operand_id", &operand_id,
          "Index of the operand to print, prints all operands otherswise."),
      tsl::Flag("output_id", &output_id,
                "Index of the output to print, prints all outputs otherswise."),
  };

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (argc < 2) {
    LOG(QFATAL) << kUsageString;
  }
  LOG(INFO) << "input file: " << argv[1];
  LOG(INFO) << "operand_id: " << operand_id;
  absl::Status s = xla::Run(argv[1], operand_id, output_id);
  if (!s.ok()) {
    std::cerr << s;
    return 1;
  }
  return 0;
}
