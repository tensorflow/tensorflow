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

#include "tensorflow/compiler/xla/tools/run_hlo_module.h"

#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/testing.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/tools/prepare_reference_module.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"

namespace xla {
namespace {

// Writes the given literal to a file in the test temporary directory.
void WriteLiteralToTempFile(const LiteralSlice& literal, const string& name) {
  // Bazel likes for tests to write "debugging outputs" like these to
  // TEST_UNDECLARED_OUTPUTS_DIR.  This plays well with tools that inspect test
  // results, especially when they're run on remote machines.
  auto* env = tensorflow::Env::Default();
  string binary_filename;
  string text_filename;
  string outdir;
  if (tensorflow::io::GetTestUndeclaredOutputsDir(&outdir)) {
    string filename = tensorflow::io::JoinPath(
        outdir, absl::StrFormat("tempfile-%d-%s", env->NowMicros(), name));
    binary_filename = absl::StrCat(filename, ".pb");
    text_filename = absl::StrCat(filename, ".txt");
  } else {
    binary_filename =
        tensorflow::io::GetTempFilename(absl::StrCat(name, ".pb"));
    text_filename = tensorflow::io::GetTempFilename(absl::StrCat(name, ".txt"));
  }

  TF_CHECK_OK(
      tensorflow::WriteBinaryProto(env, binary_filename, literal.ToProto()));
  TF_CHECK_OK(
      tensorflow::WriteStringToFile(env, text_filename, literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " binary: " << binary_filename
             << " text: " << text_filename;
}

// Callback helper that dumps literals to temporary files in the event of a
// miscomparison.
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& /*shape_index*/) {
  LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape()) << " "
            << literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape()) << " "
            << literal_comparison::ToStringTruncated(actual);
  LOG(INFO) << "Dumping literals to temp files...";
  WriteLiteralToTempFile(expected, "expected");
  WriteLiteralToTempFile(actual, "actual");
  WriteLiteralToTempFile(mismatches, "mismatches");
}

Literal ExecuteOnPlatform(std::unique_ptr<HloModule> module,
                          absl::Span<const Literal> args,
                          se::Platform* platform, bool run_hlo_passes) {
  HloRunner runner(platform);

  TF_QCHECK_OK(VerifyHloModule(module.get(), /*layout_sensitive=*/false,
                               /*allow_mixed_precision=*/true))
      << " (on " << platform->Name() << ")";

  std::cerr << "Running HLO module on platform " << platform->Name() << "...\n";
  XLA_VLOG_LINES(1, module->ToString());
  const auto start = std::chrono::high_resolution_clock::now();
  auto result_status = runner.Execute(std::move(module), args, run_hlo_passes);
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << "... compiled and ran in " << diff.count() << "s.\n";

  TF_QCHECK_OK(result_status.status())
      << "Failed to execute on " << platform->Name() << "\n";

  return result_status.ConsumeValueOrDie();
}
}  // namespace

Status RunAndCompare(
    const std::string& hlo_filename, const std::string& test_platform_name,
    const std::string& reference_platform_name, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    std::function<Status(const HloModule&,
                         const ::stream_executor::Platform::Id&, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
  se::Platform* test_platform =
      xla::PlatformUtil::GetPlatform(test_platform_name).ValueOrDie();
  se::Platform* reference_platform =
      reference_platform_name.empty()
          ? nullptr
          : xla::PlatformUtil::GetPlatform(reference_platform_name)
                .ValueOrDie();
  if (!config_modifier_hook) {
    config_modifier_hook = [](HloModuleConfig* config) {
      config->set_seed(42);
    };
  }

  std::unique_ptr<HloModule> test_module =
      LoadModuleFromFile(hlo_filename, hlo_module_loader_details::Config(),
                         options.input_format, config_modifier_hook)
          .ValueOrDie();

  if (options.flatten_control_flow) {
    HloControlFlowFlattening control_flow_flattening(
        /*while_execution_count=*/1);
    TF_RETURN_IF_ERROR(control_flow_flattening.Run(test_module.get()).status());
  }

  const HloModuleProto test_module_proto = test_module->ToProto();

  std::vector<Literal> args = MakeFakeArguments(test_module.get(), engine,
                                                options.use_large_float_range)
                                  .ConsumeValueOrDie();

  if (options.print_literals) {
    for (int i = 0; i < args.size(); ++i) {
      std::cout << "\n** Argument " << i << " **\n"
                << args[i].ToString() << "\n";
    }
  }

  std::unique_ptr<HloModule> reference_module;
  if (reference_platform != nullptr) {
    // PrepareReferenceModule needs to know the *test* platform, in order to
    // properly match the test platform's numerics.
    reference_module = PrepareReferenceModule(*test_module, test_platform->id(),
                                              config_modifier_hook,
                                              reference_module_modifier_hook)
                           .ConsumeValueOrDie();
  }

  Literal test_result = ExecuteOnPlatform(
      std::move(test_module), args, test_platform, options.run_test_hlo_passes);
  if (options.print_literals) {
    std::cout << "\n** Result on test platform " << test_platform->Name()
              << " **\n"
              << test_result.ToString() << "\n";
  }

  if (reference_module == nullptr) {
    std::cerr << "Skipping reference platform\n";
    return Status::OK();
  }

  Literal reference_result =
      ExecuteOnPlatform(std::move(reference_module), args, reference_platform,
                        options.run_reference_hlo_passes);

  if (options.print_literals) {
    std::cout << "\n** Result on reference platform "
              << reference_platform->Name() << " **\n"
              << reference_result.ToString() << "\n";
  }
  ErrorSpec error_spec(static_cast<float>(options.abs_error_bound),
                       static_cast<float>(options.rel_error_bound));
  return literal_comparison::Near(/*expected=*/reference_result,
                                  /*actual=*/test_result,
                                  /*error=*/error_spec,
                                  /*detailed_message=*/true, &OnMiscompare);
}

}  // namespace xla
