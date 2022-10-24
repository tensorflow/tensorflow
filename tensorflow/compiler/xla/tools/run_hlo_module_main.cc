/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tools/run_hlo_module.h"
#include "tensorflow/tsl/platform/init_main.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/util/command_line_flags.h"

namespace {
const char* const kUsage = R"(
This tool lets you read a HloModule from a file and execute the module on given
platform.

The file can be one of the followings:
1) a binary or text proto file, the proto should be in xla.HloProto type.
2) a hlo text dump, the string should be in HloModule::ToString() format.

By default, the module is run on a reference platform such as the interpreter
and the reference result is compared against the test result.

You can also pass in debug option flags for the HloModule.

Usage:

  bazel run run_hlo_module -- \
    --input_format=[hlo|pb|pbtxt]               \
    --platform=[CPU|CUDA|Interpreter] \
    path/to/hlo_module
)";
const char kInterpreterPlatformName[] = "Interpreter";

// Returns the name of the test platform.
std::string GetTestPlatformName(std::string name) {
  QCHECK(!name.empty()) << "Must pass --platform flag.";
  return name;
}

// Returns the name of the reference platform
std::string GetReferencePlatformName(std::string reference_platform) {
  if (reference_platform == "default") {
    return kInterpreterPlatformName;
  }
  return reference_platform;
}
}  // namespace

int main(int argc, char** argv) {
  xla::RunHloModuleOptions opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("platform", &opts.platform,
                "The test platform that the HLO module will be executed on "
                "(gpu, cpu, etc)."),
      tsl::Flag(
          "reference_platform", &opts.reference_platform,
          "The reference platform that HLO module will be "
          "executed on. The result produced on the reference platform will "
          "be compared against the result produced on the test platform. A "
          "value of 'default' will use the TPU_Interpreter as a reference if "
          "the test platform is a TPU, and 'interpreter' otherwise. If the "
          "flag value is the empty string, then the module will not be run "
          "on a reference platform at all."),
      tsl::Flag("print_literals", &opts.print_literals,
                "Print the input and result literals to stdout."),
      tsl::Flag(
          "run_test_hlo_passes", &opts.run_test_hlo_passes,
          "Run HLO pass pipeline for the test platform on the HLO module "
          "before running the module on the test platform. This should be "
          "set to true if the HLO module is unoptimized and set to false if "
          "the HLO module already has been optimized."),
      tsl::Flag(
          "run_reference_hlo_passes", &opts.run_reference_hlo_passes,
          "Run HLO pass pipeline for the reference platform on the HLO module "
          "before running the module on the reference platform. "
          "In general, if the given HLO module was optimized for a platform "
          "other "
          "than the reference this is necessary because some HLO passes are "
          "legalization passes which must be run prior to code generation."),

      tsl::Flag("use_large_float_range", &opts.use_large_float_range,
                "Generate floating point values using a large uniform-log "
                "distribution as opposed to a small uniform distribution."),
      tsl::Flag("abs_error_bound", &opts.abs_error_bound,
                "The absolute error bound used when comparing the test and "
                "reference results."),
      tsl::Flag("rel_error_bound", &opts.rel_error_bound,
                "The relative error bound used when comparing the test and "
                "reference results."),
      tsl::Flag("input_format", &opts.input_format,
                "The format of the input file. Valid values:\n"
                "  hlo : HLO textual format\n"
                "  pb : xla::HloProto in binary proto format\n"
                "  pbtxt : xla::HloProto in text proto format"),
      tsl::Flag("input_module", &opts.input_module,
                "A path to a file containing the HLO module. Can also pass "
                "a this as argv[1], but this flag is more explicit."),
      tsl::Flag(
          "iterations", &opts.iterations,
          "The number of times to run the module. Each iteration will be run "
          "with different input data.")};
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

  const std::string test_platform_name = GetTestPlatformName(opts.platform);
  const std::string reference_platform_name =
      GetReferencePlatformName(opts.reference_platform);
  auto* test_platform =
      xla::PlatformUtil::GetPlatform(test_platform_name).value();
  auto* reference_platform =
      reference_platform_name.empty()
          ? nullptr
          : xla::PlatformUtil::GetPlatform(reference_platform_name).value();
  xla::HloRunner test_runner(test_platform);
  auto reference_runner =
      reference_platform ? std::make_unique<xla::HloRunner>(reference_platform)
                         : nullptr;

  std::string hlo_filename;
  if (!opts.input_module.empty()) {
    hlo_filename = opts.input_module;
  } else {
    QCHECK(argc == 2) << "Must specify a single input file";
    hlo_filename = argv[1];
  }

  std::minstd_rand0 engine;
  int failure_count = 0;
  const int iteration_count = opts.iterations;
  for (int i = 1; i <= iteration_count; ++i) {
    if (iteration_count != 1) {
      std::cerr << "\n=== Iteration " << i << "\n";
    }
    xla::Status matched = xla::RunAndCompare(
        hlo_filename, &test_runner, reference_runner.get(), &engine, opts);

    // The AssertionResult is only meaningful when the reference is
    // used. Without a reference, the test just verifies that nothing blew up
    // when running the module.
    if (!reference_platform_name.empty()) {
      if (matched.ok()) {
        // Success.
        std::cerr << "\n** Results on " << test_platform_name << " and "
                  << reference_platform_name << " are close enough. **\n";
      } else {
        failure_count++;
        std::cerr << matched << "\n";
      }
    }
  }

  if (!reference_platform_name.empty()) {
    std::cerr << failure_count << "/" << iteration_count
              << " runs miscompared.\n";
  }

  return failure_count == 0 ? 0 : -1;
}
