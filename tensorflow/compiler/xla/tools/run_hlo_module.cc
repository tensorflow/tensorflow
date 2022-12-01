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
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/tools/prepare_reference_module.h"
#include "tensorflow/compiler/xla/tools/run_hlo_module.pb.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {
namespace {

// Writes the given literal to a file in the test temporary directory.
void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& name) {
  // Bazel likes for tests to write "debugging outputs" like these to
  // TEST_UNDECLARED_OUTPUTS_DIR.  This plays well with tools that inspect test
  // results, especially when they're run on remote machines.
  auto* env = tsl::Env::Default();
  std::string binary_filename;
  std::string text_filename;
  std::string outdir;
  if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    std::string filename = tsl::io::JoinPath(
        outdir, absl::StrFormat("tempfile-%d-%s", env->NowMicros(), name));
    binary_filename = absl::StrCat(filename, ".pb");
    text_filename = absl::StrCat(filename, ".txt");
  } else {
    binary_filename = tsl::io::GetTempFilename(absl::StrCat(name, ".pb"));
    text_filename = tsl::io::GetTempFilename(absl::StrCat(name, ".txt"));
  }

  TF_CHECK_OK(tsl::WriteBinaryProto(env, binary_filename, literal.ToProto()));
  TF_CHECK_OK(tsl::WriteStringToFile(env, text_filename, literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " binary: " << binary_filename
             << " text: " << text_filename;
}

// Callback helper that dumps literals to temporary files in the event of a
// miscomparison.
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& /*shape_index*/,
                  const literal_comparison::ErrorBuckets& /*error_buckets*/) {
  LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape()) << " "
            << literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape()) << " "
            << literal_comparison::ToStringTruncated(actual);
  LOG(INFO) << "Dumping literals to temp files...";
  WriteLiteralToTempFile(expected, "expected");
  WriteLiteralToTempFile(actual, "actual");
  WriteLiteralToTempFile(mismatches, "mismatches");
}

StatusOr<Literal> ExecuteWithRunner(std::unique_ptr<HloModule> module,
                                    absl::Span<const Literal> args,
                                    HloRunnerInterface* runner,
                                    bool run_hlo_passes) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      VerifyHloModule(module.get(), /*layout_sensitive=*/false,
                      /*allow_mixed_precision=*/true),
      absl::StrCat("(on ", runner->Name(), ")"));

  std::cerr << "Running HLO module with runner " << runner->Name() << "...\n";
  XLA_VLOG_LINES(1, module->ToString());
  const auto start = std::chrono::high_resolution_clock::now();
  auto result_status = runner->Execute(std::move(module), args, run_hlo_passes);
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << "... compiled and ran in " << diff.count() << "s.\n";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      result_status.status(),
      absl::StrCat("Failed to execute on ", runner->Name()));

  return std::move(result_status).value();
}
}  // namespace

Status RunAndCompare(
    std::unique_ptr<HloModule> test_module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
  if (!config_modifier_hook) {
    config_modifier_hook = [](HloModuleConfig* config) {
      config->set_seed(42);
    };
  }

  if (options.flatten_control_flow) {
    HloControlFlowFlattening control_flow_flattening(
        HloControlFlowFlattening::Options{/*while_execution_count=*/1});
    TF_RETURN_IF_ERROR(control_flow_flattening.Run(test_module.get()).status());
  }

  const HloModuleProto test_module_proto = test_module->ToProto();

  TF_ASSIGN_OR_RETURN(auto args,
                      MakeFakeArguments(test_module.get(), engine,
                                        options.use_large_float_range,
                                        options.treat_gte_as_data_formatting));
  // Use provided input literals as arguments, if any.
  if (iteration_literals_proto != nullptr &&
      iteration_literals_proto->arguments_size() != 0) {
    if (iteration_literals_proto->arguments_size() != args.size()) {
      return xla::InvalidArgument(
          "Failed to use input literals as arguments; mismatched "
          "number of expected arguments.");
    } else {
      for (int i = 0; i < args.size(); ++i) {
        if (!literal_comparison::EqualShapes(
                 xla::Shape(args[i].shape()),
                 xla::Shape(iteration_literals_proto->arguments(i).shape()))
                 .ok()) {
          return xla::InvalidArgument(
              "Failed to use input literals for argument %d "
              "because of a shape mismatch.",
              i);
        }
        TF_ASSIGN_OR_RETURN(args[i],
                            xla::Literal::CreateFromProto(
                                iteration_literals_proto->arguments(i)));
      }
    }
  }
  if (options.print_literals) {
    for (int i = 0; i < args.size(); ++i) {
      std::cout << "\n** Argument " << i << " **\n"
                << args[i].ToString() << "\n";
    }
  }
  if (iteration_literals_proto != nullptr &&
      iteration_literals_proto->arguments_size() == 0) {
    for (int i = 0; i < args.size(); ++i) {
      *iteration_literals_proto->add_arguments() = args[i].ToProto();
    }
  }

  std::unique_ptr<HloModule> reference_module;
  if (reference_runner != nullptr) {
    // PrepareReferenceModule needs to know the *test* runner, in order to
    // properly match the test runner's numerics.
    TF_ASSIGN_OR_RETURN(
        reference_module,
        PrepareReferenceModule(*test_module, test_runner, config_modifier_hook,
                               reference_module_modifier_hook));
  }

  TF_ASSIGN_OR_RETURN(
      auto test_result,
      ExecuteWithRunner(std::move(test_module), args, test_runner,
                        options.run_test_hlo_passes));
  if (options.print_literals) {
    std::cout << "\n** Result with test runner " << test_runner->Name()
              << " **\n"
              << test_result.ToString() << "\n";
  }
  if (iteration_literals_proto != nullptr) {
    LiteralProto test_result_proto = test_result.ToProto();
    iteration_literals_proto->mutable_result()->Swap(&test_result_proto);
  }

  if (reference_module == nullptr) {
    std::cerr << "Skipping reference runner\n";
    return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      auto reference_result,
      ExecuteWithRunner(std::move(reference_module), args, reference_runner,
                        options.run_reference_hlo_passes));

  if (options.print_literals) {
    std::cout << "\n** Result with reference runner "
              << reference_runner->Name() << " **\n"
              << reference_result.ToString() << "\n";
  }
  if (iteration_literals_proto != nullptr) {
    LiteralProto reference_result_proto = reference_result.ToProto();
    iteration_literals_proto->mutable_reference_result()->Swap(
        &reference_result_proto);
  }
  ErrorSpec error_spec(static_cast<float>(options.abs_error_bound),
                       static_cast<float>(options.rel_error_bound));
  return literal_comparison::Near(/*expected=*/reference_result,
                                  /*actual=*/test_result,
                                  /*error=*/error_spec,
                                  /*detailed_message=*/true, &OnMiscompare);
}

Status RunAndCompare(
    const std::string& hlo_filename, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
  TF_ASSIGN_OR_RETURN(
      auto test_module,
      LoadModuleFromFile(hlo_filename, hlo_module_loader_details::Config(),
                         options.input_format, config_modifier_hook));
  return RunAndCompare(std::move(test_module), test_runner, reference_runner,
                       engine, options, iteration_literals_proto,
                       reference_module_modifier_hook, config_modifier_hook);
}
}  // namespace xla
