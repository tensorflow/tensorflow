/* Copyright 2019 The OpenXLA Authors.

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

// A tool for printing compute costs. See kUsage for details.

#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/status.h"

namespace {
const char* const kUsage = R"(
This tool prints the compute cost (flops and memory traffic) of an HLO module.

The input file can be obtained from XProf graph viewer by clicking
"Download as short text".

Usage:

  bazel run compute_cost -- --input=path/to/hlo_module --format=[hlo|pb|pbtxt] [--gpu]
)";
}  // namespace

int main(int argc, char** argv) {
  std::string input, format;
  bool gpu = false;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input, "input file"),
      tsl::Flag("format", &format, "hlo|pb|pbtxt"),
      tsl::Flag("gpu", &gpu,
                "Use GPU flavor of cost analysis instead of the generic one")};
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

  TF_CHECK_OK(xla::LoadModuleFromFile(input, format, {})
                  .value()
                  ->entry_computation()
                  ->root_instruction()
                  ->Accept(&*analysis));

  std::cout << std::setw(5) << std::setprecision(4)
            << analysis->flop_count() / (1e9) << " GFLOPS. "
            << analysis->bytes_accessed() / (1e6) << " MB." << std::endl;
  return 0;
}
