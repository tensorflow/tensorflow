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

// A tool for reading a HloModule from a HloProto file and execute the module on
// given platform(s). See kUsage for details.

#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <system_error>  // NOLINT(build/c++11): required to interface with LLVM
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/translate/mhlo_to_hlo/translate.h"
#include "xla/hlo/translate/stablehlo_to_hlo/translate.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/tools/run_hlo_module.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test.h"

namespace {
const char* const kUsage = R"(
This tool lets you read a HloModule from a file and execute the module on given
platform.

The file can be one of the following:
1) An hlo text dump, the string should be in HloModule::ToString() format.
2) A binary or text proto file, the proto should be in xla.HloProto type.

By default, the module is run on a reference platform such as the interpreter
and the reference result is compared against the test result.

You can also pass in debug option flags for the HloModule.

Usage:

  bazel run run_hlo_module -- \
    --input_format=[hlo|mhlo|pb|pbtxt|stablehlo]               \
    --platform=[CPU|CUDA|Interpreter] \
    path/to/[hlo|mhlo|stablehlo]_module

Multiple files can be run as well:

  bazel run run_hlo_module -- --platform=[CPU|CUDA|Interpreter] /path/*.hlo
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
  bool different_random_seeds = false;
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
      tsl::Flag("output_literals_file", &opts.output_literals_file,
                "Output literals as RunHloModuleLiterals protobuf to the"
                " destination file."),
      tsl::Flag("input_literals_file", &opts.input_literals_file,
                "Use arguments from the provided literals file. Cannot be used "
                "in combination with \"force_fake_data\"."),
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
      tsl::Flag(
          "force_use_cpu_thunk_runtime_for_test",
          &opts.force_use_cpu_thunk_runtime_for_test,
          "Use thunk runtime for the test platform. If true, thunks runtime "
          "will be used for the test run regardless of the "
          "xla_cpu_use_thunk_runtime flag in XLA_FLAGS. This option doesn't "
          "impact reference run. It is ignored for platforms other than CPU."),
      tsl::Flag("random_init_input_literals", &opts.random_init_input_literals,
                "Initialize input literals with random numbers."
                "Leave them uninitialized otherwise."),
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
                "  mhlo : MHLO in textual or bytecode format\n"
                "  pb : xla::HloProto in binary proto format\n"
                "  pbtxt : xla::HloProto in text proto format\n"
                "  stablehlo : StableHLO in textual or bytecode format"),
      tsl::Flag(
          "iterations", &opts.iterations,
          "The number of times to run the module. Each iteration will be run "
          "with different input data."),
      tsl::Flag(
          "isolate_instructions", &opts.isolate_instructions,
          "Rather than executing the entire module at once, run every "
          "instruction individually, including the top-level and control-flow "
          "dependent computations (e.g. inside conditions, calls). Skip "
          "instructions inside fused computations etc."),
      tsl::Flag("different_random_seeds", &different_random_seeds,
                "Whether each iteration should use a different random seed for "
                "the HloModuleConfig."),
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

  QCHECK(!(opts.force_fake_data && !opts.input_literals_file.empty()))
      << "Cannot specify \"force_fake_data\" and \"input_literals_file\" "
         "together";

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

  QCHECK(argc > 1) << "Input HLO file missing.";

  int failure_count = 0;
  for (int c = 1; c < argc; c++) {
    const char* hlo_filename = argv[c];
    std::cout << "\n ** Running " << hlo_filename << "** \n";

    if (opts.input_format == "stablehlo" || opts.input_format == "mhlo") {
      auto input_filename = hlo_filename;
      hlo_filename = std::tmpnam(nullptr);

      std::error_code error;
      auto output = std::make_unique<llvm::ToolOutputFile>(
          hlo_filename, error, llvm::sys::fs::OF_None);
      if (error) {
        LOG(QFATAL) << "cannot open output file '" << std::string(hlo_filename)
                    << "': " << error.message();
      }

      auto input = llvm::MemoryBuffer::getFile(input_filename);
      error = input.getError();
      if (error) {
        LOG(QFATAL) << "cannot open input file '" << std::string(input_filename)
                    << "': " << error.message();
      }

      auto status =
          opts.input_format == "mhlo"
              ? xla::MlirHloToHloTextMain(
                    std::move(*input), output->os(),
                    /*emit_return_tuple=*/false,
                    /*emit_use_tuple_arg=*/false,
                    /*print_layouts=*/false,
                    /*print_large_constants=*/true, /*print_sugar=*/false,
                    /*via_builder=*/false, /*with_layouts=*/false)
              : xla::StablehloToHloTextMain(
                    std::move(*input), output->os(),
                    /*emit_return_tuple=*/false,
                    /*emit_use_tuple_arg=*/false,
                    /*print_layouts=*/false,
                    /*print_large_constants=*/true, /*print_sugar=*/false,
                    /*via_builder=*/false, /*with_layouts=*/false);

      if (status.failed()) {
        LOG(QFATAL) << "Failed to translate input " << opts.input_format
                    << " program to HLO text";
      }

      VLOG(1) << "Input " << opts.input_format
              << " program translated to HLO text at " << hlo_filename << "\n";

      output->keep();
      opts.input_format = "hlo";
    }

    xla::RunHloModuleLiterals literals_proto;
    std::unique_ptr<std::minstd_rand0> engine;
    if (opts.random_init_input_literals) {
      engine = std::make_unique<std::minstd_rand0>();
    }
    const int iteration_count = opts.iterations;
    xla::RunHloModuleLiterals input_literals_proto;
    if (!opts.input_literals_file.empty()) {
      ReadInputLiteralsFromFile(opts.input_literals_file,
                                &input_literals_proto);
    }

    for (int i = 0; i < iteration_count; ++i) {
      if (iteration_count != 1) {
        std::cerr << "\n=== Iteration " << i + 1 << "\n";
      }
      xla::RunHloModuleIterationLiterals* iteration_literals_proto = nullptr;
      if (!opts.output_literals_file.empty() ||
          !opts.input_literals_file.empty()) {
        iteration_literals_proto = literals_proto.add_iterations();
      }
      // If input literals are specified populate arguments portion.
      if (!opts.input_literals_file.empty() &&
          i < input_literals_proto.iterations_size()) {
        for (int argument_idx = 0;
             argument_idx < input_literals_proto.iterations(i).arguments_size();
             ++argument_idx) {
          *iteration_literals_proto->add_arguments() =
              input_literals_proto.iterations(i).arguments(argument_idx);
        }
      }
      absl::Status result = xla::RunAndCompare(
          hlo_filename, &test_runner, reference_runner.get(), engine.get(),
          opts, iteration_literals_proto,
          /*reference_module_modifier_hook=*/{},
          [&](xla::HloModuleConfig* config) {
            config->set_seed(different_random_seeds ? i + 1 : 42);
          });

      if (result.ok()) {
        if (!reference_platform_name.empty()) {
          std::cerr << "\n** Results on " << test_platform_name << " and "
                    << reference_platform_name << " are close enough. **\n";
        }
      } else {
        failure_count++;
        std::cerr << result << "\n";
      }
    }

    if (!reference_platform_name.empty()) {
      std::cerr << failure_count << "/" << iteration_count << " runs failed.\n";
    }
    if (!opts.output_literals_file.empty()) {
      if (!tsl::WriteBinaryProto(tsl::Env::Default(), opts.output_literals_file,
                                 literals_proto)
               .ok()) {
        std::cerr << "Failed to serialize literals to file "
                  << opts.output_literals_file << "\n";
      }
    }
  }

  return failure_count == 0 ? 0 : -1;
}
