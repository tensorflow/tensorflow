/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/tools/hlo_bisect/hlo_bisect_utils.h"
#include "tensorflow/tsl/platform/init_main.h"
#include "tensorflow/tsl/util/command_line_flags.h"

const char* const kUsage = R"(
Given an HloModule that manifests an XLA bug, either crashes the compiler or
an execution engine on a platform or produces observable different results on
two platforms, hlo_bisect tries to trim down the module to a smaller one that
still exhibits the same problem. We first repeatedly apply two strategies:
reduce the number of outputs produced by the entry computation and reduce
the number of sequential instructions in the entry computation. After that
we try to replace each instruction with a constant value for all the
computations to further reduce the module.

Optionally provide the --script argument in order to use an external script for
verifying the presence of the bug. This should be a path to executable that
returns a non-zero exit status if the modified HLO module (passed as the command
line argument path) has a bug.

Usage:

  bazel run hlo_bisect -- \
    --input=path/to/hlo_module \
    --test_platform=[CPU|CUDA|Interpreter]
    --dump_path=/tmp
)";

struct BisectOptions {
  std::string input = "";
  std::string script = "";
  std::string dump_path = "/tmp/hlo_bisect";
  std::string output_format = "pb";
  bool all_computations = false;
  std::string test_platform = "CUDA";
  std::string reference_platform = "Interpreter";
  float abs_error = 0.01;
  float rel_error = 0.1;
};

int main(int argc, char** argv) {
  BisectOptions opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &opts.input,
                "The input HLO module to bisect. Can also pass this as the "
                "first argument, but this flag is more explicit."),
      tsl::Flag(
          "script", &opts.script,
          "Path to the validator script. If set, then we keep reducing the HLO "
          "module as long as the script returns a non-zero exit code."),
      tsl::Flag("dump_path", &opts.dump_path,
                "The base path for dumping the HLO modules."),
      tsl::Flag("output_format", &opts.output_format,
                "The format of the output file. Valid values:\n"
                "  hlo : HLO textual format\n"
                "  pb : xla::HloProto in binary proto format"),
      tsl::Flag(
          "all_computations", &opts.all_computations,
          "Run bisection on every computation in the module. Return the "
          "innermost computation that has the bug, i.e. having the minimal "
          "module size."),
      tsl::Flag("test_platform", &opts.test_platform,
                "The platform that the HloModule will be executed on. "
                "Supported platforms: CPU, CUDA, Interpreter."),
      tsl::Flag("reference_platform", &opts.reference_platform,
                "The platform that the result will be compared against. "
                "Supported platforms are the same as test_platform."),
      tsl::Flag("abs_error", &opts.abs_error,
                "The absolute error bound used when comparing the test and "
                "reference results."),
      tsl::Flag("rel_error", &opts.rel_error,
                "The relative error bound used when comparing the test and "
                "reference results."),
  };
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
  if (opts.input.empty()) {
    QCHECK(argc == 2) << "Must specify a single input file";
    opts.input = argv[1];
  }

  auto values_or_status =
      xla::bisect::GetVerifiedModuleAndInputData(opts.input);
  if (!values_or_status.ok()) {
    LOG(ERROR) << "Failed to get verified module.";
    return 1;
  }

  std::unique_ptr<xla::HloModule> module;
  std::vector<xla::Literal> inputs;
  std::tie(module, inputs) = std::move(values_or_status).value();

  std::unique_ptr<xla::bisect::BugCheckerInterface> bug_checker;
  if (opts.script.empty()) {
    bug_checker = std::make_unique<xla::bisect::MiscompareChecker>(
        module.get(), std::move(inputs), opts.test_platform,
        opts.reference_platform,
        xla::ErrorSpec(opts.abs_error, opts.rel_error));
  } else {
    bug_checker = std::make_unique<xla::bisect::ScriptChecker>(opts.script);
  }

  auto runner = std::make_unique<xla::bisect::BisectRunner>(
      std::move(module), std::move(bug_checker));
  xla::bisect::RunBisect(std::move(runner), opts.all_computations,
                         opts.dump_path, opts.output_format);
  return 0;
}
