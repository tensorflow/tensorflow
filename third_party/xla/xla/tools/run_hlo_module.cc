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

#include "xla/tools/run_hlo_module.h"

#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/status.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_control_flow_flattening.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tools/prepare_reference_module.h"
#include "xla/tools/run_hlo_module.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
enum class ModuleResult {
  kMatched,
  kRan,
  kSkipped,
  kDidntRun,
  kOtherError,
  kCompilationError,
  kRuntimeError,
  kMismatch,
};

constexpr absl::string_view ModuleResultToString(ModuleResult result) {
  switch (result) {
    case ModuleResult::kMatched:
      return "MATCHED";
    case ModuleResult::kRan:
      return "RAN";
    case ModuleResult::kSkipped:
      return "SKIPPED";
    case ModuleResult::kDidntRun:
      return "DIDN'T RUN";
    case ModuleResult::kOtherError:
      return "OTHER ERROR";
    case ModuleResult::kCompilationError:
      return "COMPILATION ERROR";
    case ModuleResult::kRuntimeError:
      return "RUNTIME ERROR";
    case ModuleResult::kMismatch:
      return "MISMATCH";
  }
}

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

absl::StatusOr<Literal> ExecuteWithRunner(
    std::unique_ptr<HloModule> module,
    const BufferAssignmentProto* buffer_assignment_proto,
    absl::Span<const Literal> args, HloRunnerInterface* runner,
    bool run_hlo_passes) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      VerifyHloModule(module.get(), /*layout_sensitive=*/false,
                      /*allow_mixed_precision=*/true),
      absl::StrCat("(on ", runner->Name(), ")"));

  std::cerr << "Running HLO module with runner " << runner->Name() << "...\n";
  XLA_VLOG_LINES(1, module->ToString());
  const auto start = std::chrono::high_resolution_clock::now();
  ExecutionProfile profile;
  auto result_status =
      (buffer_assignment_proto == nullptr)
          ? runner->Execute(std::move(module), args, run_hlo_passes, &profile)
          : runner->ExecuteWithBufferAssignment(std::move(module),
                                                buffer_assignment_proto, args,
                                                run_hlo_passes, &profile);
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << "... compiled and ran in " << diff.count() << "s.\n";
  double run_time = static_cast<double>(profile.compute_time_ns()) / 1e9;
  std::cerr << "execution time for runner " << runner->Name() << ": "
            << run_time << "s.\n";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      result_status.status(),
      absl::StrCat("Failed to execute on ", runner->Name()));

  return std::move(result_status).value();
}

Status RunAndCompareInternal(
    std::unique_ptr<HloModule> test_module,
    const BufferAssignmentProto* buffer_assignment_proto,
    HloRunnerInterface* test_runner, HloRunnerInterface* reference_runner,
    std::minstd_rand0* engine, const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook,
    ModuleResult* test_run_result, ModuleResult* reference_run_result) {
  auto copy_result_on_failure = [](auto status, ModuleResult result,
                                   ModuleResult* out_result) {
    if (!status.ok() && out_result != nullptr) {
      *out_result = result;
    }
    return status;
  };

  if (!config_modifier_hook) {
    config_modifier_hook = [](HloModuleConfig* config) {
      config->set_seed(42);
    };
  }

  if (options.flatten_control_flow) {
    HloControlFlowFlattening control_flow_flattening(
        HloControlFlowFlattening::Options{/*while_execution_count=*/1});
    TF_RETURN_IF_ERROR(
        copy_result_on_failure(control_flow_flattening.Run(test_module.get()),
                               ModuleResult::kCompilationError, test_run_result)
            .status());
  }

  const HloModuleProto test_module_proto = test_module->ToProto();

  TF_ASSIGN_OR_RETURN(
      auto args, copy_result_on_failure(
                     MakeFakeArguments(test_module.get(), engine,
                                       options.use_large_float_range,
                                       options.treat_gte_as_data_formatting),
                     ModuleResult::kOtherError, test_run_result));
  // Use provided input literals as arguments, if any.
  if (iteration_literals_proto != nullptr &&
      iteration_literals_proto->arguments_size() != 0) {
    if (iteration_literals_proto->arguments_size() != args.size()) {
      if (test_run_result != nullptr) {
        *test_run_result = ModuleResult::kOtherError;
      }
      return xla::InvalidArgument(
          "Failed to use input literals as arguments; mismatched "
          "number of expected arguments.");
    } else {
      for (int i = 0; i < args.size(); ++i) {
        if (!literal_comparison::EqualShapes(
                 xla::Shape(args[i].shape()),
                 xla::Shape(iteration_literals_proto->arguments(i).shape()))
                 .ok()) {
          if (test_run_result != nullptr) {
            *test_run_result = ModuleResult::kOtherError;
          }
          return xla::InvalidArgument(
              "Failed to use input literals for argument %d "
              "because of a shape mismatch.",
              i);
        }
        TF_ASSIGN_OR_RETURN(
            args[i],
            copy_result_on_failure(xla::Literal::CreateFromProto(
                                       iteration_literals_proto->arguments(i)),
                                   ModuleResult::kOtherError, test_run_result));
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
        copy_result_on_failure(
            PrepareReferenceModule(*test_module, test_runner,
                                   config_modifier_hook,
                                   reference_module_modifier_hook),
            ModuleResult::kCompilationError, reference_run_result));
  }

  TF_ASSIGN_OR_RETURN(
      auto test_result,
      copy_result_on_failure(
          ExecuteWithRunner(std::move(test_module), buffer_assignment_proto,
                            args, test_runner, options.run_test_hlo_passes),
          ModuleResult::kRuntimeError, test_run_result));
  if (test_run_result != nullptr) {
    *test_run_result = ModuleResult::kRan;
  }
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
    std::cerr << "Skipping reference runner";
    return OkStatus();
  }
  if (const HloInstruction* root_instruction =
          reference_module->entry_computation()->root_instruction();
      root_instruction->opcode() == HloOpcode::kCustomCall) {
    // TODO(b/323849999) Use original computation for the reference platform.
    std::cerr << "Skipping reference runner for a custom call "
              << root_instruction->custom_call_target() << "\n";
    if (reference_run_result != nullptr) {
      *reference_run_result = ModuleResult::kSkipped;
    }
    return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      auto reference_result,
      copy_result_on_failure(
          ExecuteWithRunner(std::move(reference_module),
                            /*buffer_assignment_proto=*/nullptr, args,
                            reference_runner, options.run_reference_hlo_passes),
          ModuleResult::kRuntimeError, reference_run_result));
  if (reference_run_result != nullptr) {
    *reference_run_result = ModuleResult::kRan;
  }

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

  Status comparison_status =
      literal_comparison::Near(/*expected=*/reference_result,
                               /*actual=*/test_result,
                               /*error=*/error_spec,
                               /*detailed_message=*/true, &OnMiscompare);
  const ModuleResult comparison_result =
      comparison_status.ok() ? ModuleResult::kMatched : ModuleResult::kMismatch;
  if (test_run_result != nullptr) {
    *test_run_result = comparison_result;
  }
  if (reference_run_result != nullptr) {
    *reference_run_result = comparison_result;
  }
  return comparison_status;
}

struct ChunkResult {
  std::string module_name;
  ModuleResult test_result = ModuleResult::kDidntRun;
  ModuleResult reference_result = ModuleResult::kDidntRun;
  Status status;

  bool operator<(const ChunkResult& other) const {
    if (test_result != other.test_result) {
      return test_result < other.test_result;
    }
    return reference_result < other.reference_result;
  }
};

std::string BuildResultsTable(absl::Span<const ChunkResult> chunk_results,
                              size_t num_modules) {
  constexpr int kStatusWidth = 21;
  constexpr int kNameWidth = 30;
  constexpr int kThreeColumnsWidth = 5 + 2 * kStatusWidth + kNameWidth;
  constexpr int kTableWidth = kThreeColumnsWidth + 30;

  std::ostringstream strstr;
  auto print_row = [&](absl::string_view reference, absl::string_view test,
                       absl::string_view module_name, absl::string_view error) {
    std::string formatted_error = absl::StrReplaceAll(
        error, {{"\n", absl::StrCat("\n", std::string(kThreeColumnsWidth, ' '),
                                    "|")}});
    strstr << " " << std::left << std::setw(kStatusWidth) << reference << "| "
           << std::setw(kStatusWidth) << test << "| " << std::setw(kNameWidth)
           << module_name << "| " << formatted_error << "\n";
  };
  auto print_line = [&](int line_width) {
    strstr << std::string(line_width, '-') << "\n";
  };

  print_row("Reference", "Test", "Module", "Status");
  print_line(kTableWidth);

  std::map<std::pair<ModuleResult, ModuleResult>, int> result_counts;

  for (const ChunkResult& chunk_result : chunk_results) {
    const std::pair<ModuleResult, ModuleResult> result_pair(
        chunk_result.reference_result, chunk_result.test_result);

    ++result_counts[result_pair];
    print_row(ModuleResultToString(chunk_result.reference_result),
              ModuleResultToString(chunk_result.test_result),
              chunk_result.module_name, chunk_result.status.ToString());
  }
  print_line(kTableWidth);
  print_row("Reference", "Test", "Module", "Status");
  print_line(kTableWidth);

  strstr << "\n\n";

  // Summary table.
  print_line(kThreeColumnsWidth);
  print_row("Reference", "Test", "Total count", "");
  print_line(kThreeColumnsWidth);
  for (const auto& [result, count] : result_counts) {
    print_row(ModuleResultToString(result.first),
              ModuleResultToString(result.second), absl::StrCat(count), "");
  }
  print_line(kThreeColumnsWidth);
  if (chunk_results.size() < num_modules) {
    strstr << "\n(did not " << (num_modules - chunk_results.size())
           << " modules due to earlier failures)\n\n";
  }
  return strstr.str();
}

Status RunIsolatedAndCompare(
    std::unique_ptr<HloModule> test_module,
    const BufferAssignmentProto* buffer_assignment_proto,
    HloRunnerInterface* test_runner, HloRunnerInterface* reference_runner,
    std::minstd_rand0* engine, const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
  CHECK(test_module);
  CHECK(iteration_literals_proto == nullptr)
      << "Cannot run decomposed module if input literals are provided.";
  if (options.run_test_hlo_passes || (options.run_reference_hlo_passes &&
                                      !options.reference_platform.empty())) {
    LOG(WARNING)
        << "!!! Warning !!! When running decomposed module, running HLO "
           "passes is likely not what you want. If you have unoptimized "
           "HLO, first convert it to the optimized e.g. using the "
           "hlo-opt tool, and then isolate without HLO passes.";
  }

  std::vector<ChunkResult> chunk_results;

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<HloModule>> modules,
      DecomposeHloModule(*test_module, /*deduplicate_modules=*/true));

  Status status = OkStatus();
  for (std::unique_ptr<HloModule>& module : modules) {
    const std::string module_name = module->name();
    ModuleResult test_module_result = ModuleResult::kDidntRun;
    ModuleResult reference_module_result = ModuleResult::kDidntRun;
    Status chunk_status = RunAndCompareInternal(
        std::move(module), buffer_assignment_proto, test_runner,
        reference_runner, engine, options, iteration_literals_proto,
        reference_module_modifier_hook, config_modifier_hook,
        &test_module_result, &reference_module_result);
    chunk_results.push_back({std::move(module_name), test_module_result,
                             reference_module_result, chunk_status});
    status.Update(chunk_status);
    if (!chunk_status.ok() && test_module_result != ModuleResult::kMismatch) {
      break;
    }
  }
  absl::c_sort(chunk_results);
  std::cout << BuildResultsTable(chunk_results, modules.size());
  return status;
}

}  // namespace

Status RunAndCompare(
    std::unique_ptr<HloModule> test_module,
    const BufferAssignmentProto* buffer_assignment_proto,
    HloRunnerInterface* test_runner, HloRunnerInterface* reference_runner,
    std::minstd_rand0* engine, const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
  if (options.isolate_instructions) {
    return RunIsolatedAndCompare(
        std::move(test_module), buffer_assignment_proto, test_runner,
        reference_runner, engine, options, iteration_literals_proto,
        reference_module_modifier_hook, config_modifier_hook);
  }
  return RunAndCompareInternal(
      std::move(test_module), buffer_assignment_proto, test_runner,
      reference_runner, engine, options, iteration_literals_proto,
      reference_module_modifier_hook, config_modifier_hook, nullptr, nullptr);
}

Status RunAndCompare(
    const std::string& hlo_filename, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook,
    std::function<Status(const RunHloModuleOptions& options, HloModule& module)>
        compilation_env_modifier_hook) {
  BufferAssignmentProto buffer_assignment_proto;
  TF_ASSIGN_OR_RETURN(
      auto test_module,
      LoadModuleFromFile(
          hlo_filename, options.input_format,
          hlo_module_loader_details::Config(), config_modifier_hook,
          options.use_buffer_assignment_from_proto ? &buffer_assignment_proto
                                                   : nullptr));
  HloVerifier verifier(
      HloVerifierOpts{}.WithLayoutSensitive(false).WithAllowMixedPrecision(
          true));
  TF_RETURN_IF_ERROR(verifier.Run(test_module.get()).status());
  if (compilation_env_modifier_hook) {
    TF_CHECK_OK(compilation_env_modifier_hook(options, *test_module))
        << "Could not adjust the compilation environment for user provided "
           "hlo module.";
  }

  if (options.print_literals) {
    std::cout << "\n** Buffer assignment proto **\n"
              << buffer_assignment_proto.DebugString() << "\n";
  }
  std::unique_ptr<RunHloModuleIterationLiterals> iteration_literals_proto_local;
  if (iteration_literals_proto == nullptr) {
    // User did not explicitly give input
    if (!options.force_fake_data && !options.isolate_instructions &&
        (options.input_format == "pb" || options.input_format == "pbtxt")) {
      // User is giving a snapshot (which contains inputs)
      LOG(INFO) << "Using input data from the user-provided snapshot.";
      TF_ASSIGN_OR_RETURN(
          iteration_literals_proto_local,
          LoadInputFromFile(hlo_filename, options.input_format));
      iteration_literals_proto = iteration_literals_proto_local.get();
    } else if (options.input_format == "pb" ||
               options.input_format == "pbtxt") {
      LOG(INFO)
          << "Ignoring input data from snapshot and using fake data instead.";
    }
  }
  return RunAndCompare(
      std::move(test_module),
      options.use_buffer_assignment_from_proto ? &buffer_assignment_proto
                                               : nullptr,
      test_runner, reference_runner, engine, options, iteration_literals_proto,
      reference_module_modifier_hook, config_modifier_hook);
}
}  // namespace xla
