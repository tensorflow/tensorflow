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

// A tool for printing benchmark entries for CPU backend's dot_benchmark_test.
// See kUsage for details.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/protobuf.h"

namespace {
const char* const kUsage = R"(
This tool prints all dots of an HLO module in a format that can be pasted into
//xla/backends/cpu/benchmarks/dot_benchmark_test.cc.

Usage:

  bazel run extract_dots_for_benchmark -- --input=path/to/hlo_module \
    --format=[hlo|pb|pbtxt]

Example output:
  GenericDot{name, BF16, {1,11,1152}, BF16, {4,1152,256}, BF16, {1,11,4,256}, {}, {}, {2}, {1}},
  GenericDot{name, BF16, {2,1,1152,256}, BF16, {1,11,1152}, BF16, {2,1,256,1,11}, {}, {}, {2}, {2}},
  GenericDot{name, BF16, {1,11,4,256}, BF16, {1,11,256}, BF16, {1,11,4,11}, {0}, {0}, {3}, {2}},
)";
}  // namespace

namespace xla {
namespace {

std::string ShapeToBenchmarkString(const Shape& shape) {
  return absl::StrCat(
      absl::AsciiStrToUpper(
          primitive_util::LowercasePrimitiveTypeName(shape.element_type())),
      ", {", absl::StrJoin(shape.dimensions(), ","), "}");
}

std::string TupleToString(const tsl::protobuf::RepeatedField<int64_t>& tuple) {
  std::vector<int64_t> tuple_vec(tuple.begin(), tuple.end());
  return absl::StrCat("{", absl::StrJoin(tuple_vec, ","), "}");
}

void PrintDots(const HloModule& module) {
  absl::flat_hash_set<std::string> entries;  // Only print unique dots.
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kDot) {
        DotDimensionNumbers dot_dims = hlo->dot_dimension_numbers();
        std::string entry = absl::StrCat(
            "GenericDot{name, ",
            ShapeToBenchmarkString(hlo->operand(0)->shape()), ", ",
            ShapeToBenchmarkString(hlo->operand(1)->shape()), ", ",
            ShapeToBenchmarkString(hlo->shape()), ", ",
            TupleToString(dot_dims.lhs_batch_dimensions()), ", ",
            TupleToString(dot_dims.rhs_batch_dimensions()), ", ",
            TupleToString(dot_dims.lhs_contracting_dimensions()), ", ",
            TupleToString(dot_dims.rhs_contracting_dimensions()), "}");
        entries.insert(entry);
      }
    }
  }

  // Print entries in alphabetical order.
  std::vector<absl::string_view> entries_vec(entries.begin(), entries.end());
  std::sort(entries_vec.begin(), entries_vec.end());
  for (auto& entry : entries_vec) {
    std::cout << entry << ",\n";
  }
}
}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::string input, format;
  bool gpu = false;
  bool all = false;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input, "input file"),
      tsl::Flag("format", &format, "hlo|pb|pbtxt"),
      tsl::Flag("gpu", &gpu,
                "Use GPU flavor of cost analysis instead of the generic one"),
      tsl::Flag(
          "all", &all,
          "Also print costs and deduplicated name of each instruction, not "
          "just the total costs for the module")};
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  std::unique_ptr<xla::HloCostAnalysis> analysis;
  if (gpu) {
    analysis = std::make_unique<xla::gpu::GpuHloCostAnalysis>(
        xla::HloCostAnalysis::Options{});
  } else {
    analysis = std::make_unique<xla::HloCostAnalysis>();
  }

  std::unique_ptr<xla::HloModule> module =
      *xla::LoadModuleFromFile(input, format, {});

  TF_CHECK_OK(
      module->entry_computation()->root_instruction()->Accept(&*analysis));

  xla::PrintDots(*module);

  return 0;
}
