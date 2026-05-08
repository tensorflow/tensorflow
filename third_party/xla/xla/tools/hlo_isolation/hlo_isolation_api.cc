/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tools/hlo_isolation/hlo_isolation_api.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "xla/comparison_util.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/defuser.h"
#include "xla/hlo/transforms/despecializer.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

using ::xla::hlo_isolation::ModuleIsolationOptions;
using ::xla::hlo_isolation::PipelineIsolationOptions;

namespace xla {
namespace hlo_isolation {

namespace {

absl::Status InitIsolatorOptions(ModuleIsolationOptions& options) {
  if (!options.run_module_fn) {
    options.run_module_fn =
        [run_hlo_passes = options.run_hlo_passes](
            std::unique_ptr<HloModule> m, HloRunnerInterface* r,
            absl::Span<const Literal> i) -> absl::StatusOr<Literal> {
      return RunModule(std::move(m), r, i, run_hlo_passes);
    };
  }
  if (!options.on_mismatch_fn) {
    options.on_mismatch_fn = [](const HloModule& module,
                                const Literal& /*test_output*/,
                                const Literal& /*reference_output*/,
                                const absl::Status& /*compare_status*/) {
      auto* env = tsl::Env::Default();
      std::string outdir;
      std::string filename;
      if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
        filename = tsl::io::JoinPath(
            outdir, absl::StrCat("failed_module-", env->NowMicros(), "-",
                                 module.name(), ".txt"));
      } else {
        filename = tsl::io::GetTempFilename(
            absl::StrCat("failed_module-", module.name(), ".txt"));
      }
      CHECK_OK(tsl::WriteStringToFile(env, filename, module.ToString()));
      LOG(INFO) << "Wrote failed HLO module to " << filename;
    };
  }
  if (!options.make_fake_arguments_fn) {
    options.make_fake_arguments_fn =
        [](const HloModule& module) -> absl::StatusOr<std::vector<Literal>> {
      return MakeFakeArguments(&module);
    };
  }
  if (!options.estimate_module_size_fn) {
    options.estimate_module_size_fn = [](const HloModule& module) -> int64_t {
      int64_t total_size = 0;
      for (const auto* param :
           module.entry_computation()->parameter_instructions()) {
        total_size += ShapeUtil::ByteSizeOf(param->shape());
      }
      total_size += ShapeUtil::ByteSizeOf(
          module.entry_computation()->root_instruction()->shape());
      return total_size;
    };
  }
  return absl::OkStatus();
}

absl::Status ValidatePipelineOptions(const PipelineIsolationOptions& options) {
  if (options.shard_index >= 0) {
    if (options.num_shards <= 0) {
      return absl::InvalidArgumentError(
          "num_shards must be strictly positive when shard_index is "
          "specified.");
    }
    if (options.shard_index >= options.num_shards) {
      return absl::InvalidArgumentError(
          "shard_index must be less than num_shards.");
    }
  }
  return absl::OkStatus();
}

void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& module_name,
                            const std::string& name) {
  auto* env = tsl::Env::Default();
  std::string binary_filename;
  std::string text_filename;
  std::string outdir;
  std::string prefix = absl::StrCat(module_name, "-", name);
  if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    std::string filename = tsl::io::JoinPath(
        outdir, absl::StrCat("failed-", env->NowMicros(), "-", prefix));
    binary_filename = absl::StrCat(filename, ".pb");
    text_filename = absl::StrCat(filename, ".txt");
  } else {
    binary_filename = tsl::io::GetTempFilename(absl::StrCat(prefix, ".pb"));
    text_filename = tsl::io::GetTempFilename(absl::StrCat(prefix, ".txt"));
  }

  CHECK_OK(tsl::WriteBinaryProto(env, binary_filename, literal.ToProto()));
  CHECK_OK(tsl::WriteStringToFile(env, text_filename, literal.ToString()));
  LOG(INFO) << "Wrote Literal to " << prefix << " binary: " << binary_filename
            << " text: " << text_filename;
}

absl::Status CompareOutputs(const HloModule& module, const Literal& test_output,
                            const Literal& reference_output,
                            HloIsolationTestResult& result,
                            const ModuleIsolationOptions& options,
                            absl::string_view check_name) {
  ErrorSpec error_spec(options.abs_error_bound, options.rel_error_bound);
  auto on_miscompare =
      [&module](const LiteralSlice& expected, const LiteralSlice& actual,
                const LiteralSlice& mismatches, const ShapeIndex& shape_index,
                const literal_comparison::ErrorBuckets& /*error_buckets*/) {
        std::string escaped_shape_index = absl::StrReplaceAll(
            shape_index.ToString(), {{",", "_"}, {"{", ""}, {"}", ""}});
        std::string shape_suffix =
            escaped_shape_index.empty()
                ? ""
                : absl::StrCat("-shape-", escaped_shape_index);
        WriteLiteralToTempFile(expected, module.name(),
                               absl::StrCat("expected", shape_suffix));
        WriteLiteralToTempFile(actual, module.name(),
                               absl::StrCat("actual", shape_suffix));
        WriteLiteralToTempFile(mismatches, module.name(),
                               absl::StrCat("mismatches", shape_suffix));
      };
  absl::Status status = literal_comparison::Near(
      reference_output, test_output, error_spec, true, on_miscompare);
  NumericCheck* numeric_check = result.add_numeric_checks();
  numeric_check->set_name(check_name);
  numeric_check->set_expected_contains_inf_or_nan(
      LiteralContainsInfOrNan(reference_output));
  numeric_check->set_actual_contains_inf_or_nan(
      LiteralContainsInfOrNan(test_output));
  if (!status.ok()) {
    status = absl::InternalError(
        absl::StrFormat("Value mismatch in check %s for module %s\n\n%s",
                        check_name, module.name(), status.message()));
    options.on_mismatch_fn(module, test_output, reference_output, status);
    absl::StatusOr<std::vector<NumericMismatch>> top_mismatches =
        ExtractAndEnrichTopMismatches(std::string(status.message()), &module);
    if (!top_mismatches.ok()) {
      LOG(ERROR) << "Failed to extract top relative error mismatch: "
                 << top_mismatches.status();
    } else if (top_mismatches->empty()) {
      LOG(ERROR) << "No top relative error mismatches found.";
    } else {
      numeric_check->clear_top_mismatches();
      for (const NumericMismatch& mismatch : *top_mismatches) {
        *numeric_check->add_top_mismatches() = mismatch;
      }
      NumericMismatch top_relative_error_mismatch = top_mismatches->at(0);
      for (const NumericMismatch& mismatch : *top_mismatches) {
        if (mismatch.rel_error() > top_relative_error_mismatch.rel_error()) {
          top_relative_error_mismatch = mismatch;
        }
      }
      *numeric_check->mutable_top_mismatch() = top_relative_error_mismatch;
    }
  }
  return status;
}

}  // namespace

absl::StatusOr<Literal> RunModule(std::unique_ptr<HloModule> module,
                                  HloRunnerInterface* runner,
                                  absl::Span<const Literal> input_data,
                                  bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(auto executable, runner->CreateExecutable(
                                           std::move(module), run_hlo_passes));
  return runner->ExecuteWithExecutable(executable.get(), input_data);
}

absl::StatusOr<HloIsolationTestResult> RunIsolationTestOnModule(
    const HloModule& module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, ModuleIsolationOptions options,
    absl::Span<const Literal> input_data) {
  HloIsolationTestResult result;
  result.set_module_name(module.name());

  TF_RETURN_IF_ERROR(InitIsolatorOptions(options));

  std::vector<Literal> local_inputs;
  if (input_data.empty()) {
    TF_ASSIGN_OR_RETURN(local_inputs, options.make_fake_arguments_fn(module));
    input_data = local_inputs;
  }

  // Run a series of checks on the module. If any one of them passes, consider
  // the module to be correct immediately.
  // 1. TPU vs defused TPU
  // 2. TPU vs interpreter
  // 3. Try another input (e.g., uniform(0.1, 1))
  //   - Repeat TPU vs defused TPU
  //   - Repeat TPU vs interpreter

  auto run_module = [&](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
                        absl::Span<const Literal> i) {
    return options.run_module_fn(std::move(m), r, i);
  };

  // Run test runner.
  absl::StatusOr<Literal> test_output =
      run_module(module.Clone(""), test_runner, input_data);
  if (!test_output.ok()) {
    result.set_state(State::FAILURE);
    result.set_reason("TEST_RUNNER_FAILURE");
    LOG(ERROR) << "Test runner failed: " << test_output.status()
               << " for module: " << module.name();
    return result;
  }

  // Run defused test runner.
  std::unique_ptr<HloModule> defused_module = module.Clone("defused");
  TF_RETURN_IF_ERROR(DefuseModule(defused_module.get()));
  absl::StatusOr<Literal> defused_output =
      run_module(std::move(defused_module), test_runner, input_data);
  if (!defused_output.ok()) {
    result.set_state(State::FAILURE);
    result.set_reason("DEFUSED_TEST_RUNNER_FAILURE");
    LOG(ERROR) << "Test runner failed for defused module: "
               << defused_output.status() << " for module: " << module.name();
    return result;
  }

  // Compare Test vs Defused Test.
  absl::Status compare_status =
      CompareOutputs(module, *test_output, *defused_output, result, options,
                     "TPU_VS_DEFUSED_TPU");
  if (compare_status.ok()) {
    result.set_state(State::SUCCESS);
    result.set_reason("STAGE_1_DEFUSED_TPU_SUCCESS");
    return result;
  }

  // Potentially skip reference run.
  if (options.max_module_size_bytes > 0) {
    int64_t size = options.estimate_module_size_fn(module);
    if (size > options.max_module_size_bytes) {
      LOG(INFO) << "Skipping reference run for module: " << module.name()
                << " due to large size: " << size;
      reference_runner = nullptr;
    }
  }
  if (reference_runner != nullptr && ModuleContainsLargeKeyValueSort(module)) {
    LOG(INFO) << "Skipping reference run for module: " << module.name()
              << " due to large key value sort";
    reference_runner = nullptr;
  }

  if (reference_runner) {
    std::unique_ptr<HloModule> despecialized_module =
        module.Clone("despecialized");
    Despecializer despecializer;
    TF_RETURN_IF_ERROR(despecializer.Run(despecialized_module.get()).status());
    std::string despecialized_module_name = despecialized_module->name();

    // Run the reference runner.
    absl::StatusOr<Literal> reference_output = run_module(
        std::move(despecialized_module), reference_runner, input_data);
    if (!reference_output.ok()) {
      result.set_state(State::FAILURE);
      result.set_reason("REFERENCE_RUNNER_FAILURE");
      LOG(ERROR) << "Reference runner failed: " << reference_output.status()
                 << " for module: " << despecialized_module_name;
      return result;
    }

    // Compare Test vs Reference.
    absl::Status compare_status =
        CompareOutputs(module, *test_output, *reference_output, result, options,
                       "TPU_VS_INTERPRETER");
    if (compare_status.ok()) {
      result.set_state(State::SUCCESS);
      result.set_reason("STAGE_2_INTERPRETER_SUCCESS");
      return result;
    }
  }

  result.set_state(State::FAILURE);
  result.set_reason("NUMERIC_MISMATCH");
  return result;
}

absl::StatusOr<std::vector<HloIsolationTestResult>> RunIsolationPipeline(
    const HloModule& input_module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, PipelineIsolationOptions options) {
  TF_RETURN_IF_ERROR(ValidatePipelineOptions(options));
  TF_RETURN_IF_ERROR(InitIsolatorOptions(options.module_options));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<HloModule>> modules,
      DecomposeHloModule(input_module, /*deduplicate_modules=*/true));

  // Sort submodules by name and fingerprint to ensure deterministic sharding.
  std::sort(modules.begin(), modules.end(),
            [](const std::unique_ptr<HloModule>& a,
               const std::unique_ptr<HloModule>& b) {
              if (a->name() != b->name()) {
                return a->name() < b->name();
              }
              return a->GetFingerprint128() < b->GetFingerprint128();
            });

  std::vector<HloIsolationTestResult> pipeline_results;
  int64_t filtered_module_index = 0;
  for (int i = 0; i < modules.size(); ++i) {
    auto& module = modules[i];

    bool is_filtered = false;
    std::string skip_reason;
    if (!options.filter_by_name.empty() &&
        !RE2::FullMatch(module->name(), options.filter_by_name)) {
      is_filtered = true;
      skip_reason = "NO_MATCH_FILTER_BY_NAME";
    } else if (!options.skip_by_name.empty() &&
               RE2::FullMatch(module->name(), options.skip_by_name)) {
      is_filtered = true;
      skip_reason = "MATCH_SKIP_BY_NAME";
    } else {
      bool has_matching_opcode = false;
      bool has_skipped_opcode = false;
      for (const auto* computation : module->computations()) {
        for (const auto* instruction : computation->instructions()) {
          std::string opcode_str(HloOpcodeString(instruction->opcode()));
          if (options.filter_by_opcode.empty() ||
              RE2::FullMatch(opcode_str, options.filter_by_opcode)) {
            has_matching_opcode = true;
          }
          if (!options.skip_by_opcode.empty() &&
              RE2::FullMatch(opcode_str, options.skip_by_opcode)) {
            has_skipped_opcode = true;
          }
        }
      }
      if (!has_matching_opcode || has_skipped_opcode) {
        is_filtered = true;
        skip_reason = "NO_MATCH_FILTER_BY_OPCODE";
        if (has_skipped_opcode) {
          skip_reason = "MATCH_SKIP_BY_OPCODE";
        }
      }
    }

    if (is_filtered) {
      LOG(INFO) << "Module " << module->name() << " skipped: " << skip_reason;
      HloIsolationTestResult skipped_result;
      skipped_result.set_module_name(module->name());
      skipped_result.set_state(State::SKIPPED);
      skipped_result.set_reason(skip_reason);
      pipeline_results.push_back(std::move(skipped_result));
      continue;
    }

    // Sharding check happens after all filters are cleared
    if (options.shard_index >= 0 && options.num_shards > 0) {
      if (filtered_module_index % options.num_shards != options.shard_index) {
        filtered_module_index++;
        continue;
      }
      filtered_module_index++;
    }

    // Execute module
    absl::StatusOr<HloIsolationTestResult> result_or = RunIsolationTestOnModule(
        *module, test_runner, reference_runner, options.module_options);
    if (!result_or.ok()) {
      LOG(ERROR) << "Failed to run isolation test on module " << module->name()
                 << ": " << result_or.status();
      HloIsolationTestResult failed_result;
      failed_result.set_module_name(module->name());
      failed_result.set_state(State::FAILURE);
      failed_result.set_reason(result_or.status().message());
      pipeline_results.push_back(std::move(failed_result));
      continue;
    }
    pipeline_results.push_back(std::move(*result_or));
  }

  return pipeline_results;
}

absl::StatusOr<std::vector<HloIsolationTestResult>> RunIsolationPipeline(
    const std::string& input_path, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, PipelineIsolationOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> loaded_module,
                      LoadModuleFromFile(input_path));
  return RunIsolationPipeline(*loaded_module, test_runner, reference_runner,
                              options);
}

absl::Status DefuseModule(HloModule* module) {
  HloPassPipeline pipeline("defuser");
  pipeline.AddPass<HloDescheduler>();
  pipeline.AddPass<Defuser>();
  pipeline.AddPass<HloTrivialScheduler>();
  return pipeline.Run(module).status();
}

absl::StatusOr<NumericMismatch> ParseMismatchLine(absl::string_view line) {
  std::string actual_str, expected_str, index_str, rel_error_str, abs_error_str;
  if (RE2::PartialMatch(
          line,
          R"(actual\s+([^,]+),\s+expected\s+([^,]+),\s+index\s+(.+?),\s+rel error\s+([^,]+),\s+abs error\s+(.+))",
          &actual_str, &expected_str, &index_str, &rel_error_str,
          &abs_error_str)) {
    double actual_double, expected_double, rel_error_double;
    if (!absl::SimpleAtod(actual_str, &actual_double) ||
        !absl::SimpleAtod(expected_str, &expected_double) ||
        !absl::SimpleAtod(rel_error_str, &rel_error_double)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse numeric values from line: ", line));
    }
    NumericMismatch data;
    data.set_actual(actual_double);
    data.set_expected(expected_double);
    data.set_rel_error(rel_error_double);
    return data;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Failed to match line: ", line));
}

absl::StatusOr<std::vector<NumericMismatch>> ExtractTopMismatches(
    std::string error_message, bool is_tuple) {
  std::stringstream ss(error_message);
  std::string line;
  std::vector<NumericMismatch> mismatches;
  std::optional<int64_t> shape_index;
  if (!is_tuple) {
    shape_index = 0;
  }
  std::optional<NumericMismatch> current_mismatch;
  std::optional<double> parsed_mismatch_percentage;

  bool parsed_abs_percentage = false;
  bool parsed_rel_percentage = false;

  while (std::getline(ss, line)) {
    if (!shape_index.has_value()) {
      std::string parsed_shape_index_str;
      if (RE2::PartialMatch(line, R"(Array at shape index\s*\{\s*(\d+))",
                            &parsed_shape_index_str)) {
        int64_t idx;
        if (!absl::SimpleAtoi(parsed_shape_index_str, &idx)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Failed to parse shape index from line: ", line));
        }
        shape_index = idx;
        continue;
      }
    }

    if (!parsed_mismatch_percentage.has_value()) {
      std::string parsed_mismatch_percentage_str;
      if (RE2::PartialMatch(line, R"(Mismatch count\s*\d+\s*\(([^%]+)%\))",
                            &parsed_mismatch_percentage_str)) {
        double percentage;
        if (!absl::SimpleAtod(parsed_mismatch_percentage_str, &percentage)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Failed to parse mismatch percentage from line: ", line));
        }
        parsed_mismatch_percentage = percentage;
        continue;
      }
    }

    absl::StatusOr<NumericMismatch> parsed = ParseMismatchLine(line);
    if (!current_mismatch.has_value() && parsed.ok()) {
      CHECK(shape_index.has_value());
      parsed->set_output_shape_index(*shape_index);
      current_mismatch = std::move(*parsed);
      current_mismatch->set_percentage_of_elems_exceeding_both_errors(
          parsed_mismatch_percentage.value_or(0.0));
      continue;
    }

    std::string percentage_str;
    if (!parsed_abs_percentage &&
        RE2::PartialMatch(
            line,
            R"(Elements exceeding abs error bound[^:]*:\s*\d+\s*\(([^%]+)%\))",
            &percentage_str)) {
      CHECK(current_mismatch.has_value());
      double percentage;
      if (absl::SimpleAtod(percentage_str, &percentage)) {
        current_mismatch->set_percentage_of_elems_exceeding_abs_error(
            percentage);
        parsed_abs_percentage = true;
      }
    } else if (
        !parsed_rel_percentage &&
        RE2::PartialMatch(
            line,
            R"(Elements exceeding rel error bound[^:]*:\s*\d+\s*\(([^%]+)%\))",
            &percentage_str)) {
      CHECK(current_mismatch.has_value());
      double percentage;
      if (absl::SimpleAtod(percentage_str, &percentage)) {
        current_mismatch->set_percentage_of_elems_exceeding_rel_error(
            percentage);
        parsed_rel_percentage = true;
      }
    }

    if (current_mismatch.has_value() && parsed_abs_percentage &&
        parsed_rel_percentage && parsed_mismatch_percentage.has_value()) {
      mismatches.push_back(std::move(*current_mismatch));
      parsed_abs_percentage = false;
      parsed_rel_percentage = false;
      parsed_mismatch_percentage = std::nullopt;
      shape_index = std::nullopt;
      current_mismatch = std::nullopt;
    }
  }
  if (current_mismatch.has_value()) {
    mismatches.push_back(std::move(*current_mismatch));
  }
  return mismatches;
}

absl::StatusOr<NumericMismatch> ExtractTopRelativeErrorMismatch(
    std::string error_message) {
  TF_ASSIGN_OR_RETURN(std::vector<NumericMismatch> mismatches,
                      ExtractTopMismatches(error_message, false));
  if (mismatches.empty()) {
    return absl::NotFoundError(
        "Could not find top relative error mismatch in the error message.");
  }
  NumericMismatch top_relative_error_mismatch = mismatches.front();
  for (const auto& mismatch : mismatches) {
    if (mismatch.rel_error() > top_relative_error_mismatch.rel_error()) {
      top_relative_error_mismatch = mismatch;
    }
  }
  return top_relative_error_mismatch;
}

absl::StatusOr<std::vector<bool>> DetectReducesInModuleOutput(
    const HloModule* module) {
  const Shape& output_shape = module->result_shape();
  int64_t num_outputs = 1;
  if (output_shape.IsTuple()) {
    num_outputs = output_shape.tuple_shapes().size();
  }
  std::vector<bool> reduce_in_output(num_outputs, false);
  std::unique_ptr<HloModule> defused_module = module->Clone("defused");
  TF_RETURN_IF_ERROR(DefuseModule(defused_module.get()));

  auto bfs = [&reduce_in_output](HloModule* module,
                                 int64_t output_index) -> void {
    absl::flat_hash_set<const HloInstruction*> visited;
    std::queue<const HloInstruction*> q;
    if (module->result_shape().IsTuple()) {
      if (module->entry_computation()->root_instruction()->operands().size() >
          output_index) {
        q.push(module->entry_computation()->root_instruction()->operand(
            output_index));
      }
    } else {
      q.push(module->entry_computation()->root_instruction());
    }
    while (!q.empty()) {
      const HloInstruction* current = q.front();
      q.pop();
      if (visited.contains(current)) {
        continue;
      }
      visited.insert(current);
      if (current->opcode() == HloOpcode::kReduce) {
        reduce_in_output[output_index] = true;
      }
      for (const HloInstruction* operand : current->operands()) {
        if (operand->opcode() == HloOpcode::kGetTupleElement) {
          int64_t tuple_index = operand->tuple_index();
          const HloInstruction* tuple = operand->operand(0);
          if (tuple->operands().size() > tuple_index) {
            const HloInstruction* tuple_element = tuple->operand(tuple_index);
            visited.insert(tuple);
            visited.insert(operand);
            q.push(tuple_element);
          }
        } else {
          q.push(operand);
        }
      }
    }
  };

  for (int64_t i = 0; i < num_outputs; ++i) {
    bfs(defused_module.get(), i);
  }
  return reduce_in_output;
}

absl::StatusOr<std::vector<NumericMismatch>> ExtractAndEnrichTopMismatches(
    std::string error_message, const HloModule* module) {
  bool is_tuple = module->result_shape().IsTuple();
  int64_t num_outputs =
      is_tuple ? module->result_shape().tuple_shapes().size() : 1;

  TF_ASSIGN_OR_RETURN(std::vector<NumericMismatch> mismatches,
                      ExtractTopMismatches(error_message, is_tuple));
  TF_ASSIGN_OR_RETURN(std::vector<bool> reduce_in_output,
                      DetectReducesInModuleOutput(module));
  for (NumericMismatch& mismatch : mismatches) {
    int output_index = mismatch.output_shape_index();
    if (output_index >= num_outputs) {
      return absl::InternalError(
          absl::StrCat("Invalid output index: ", output_index));
    }
    mismatch.set_result_of_reduce(reduce_in_output[output_index]);
  }
  return mismatches;
}

int64_t GetFusionCountInNestedFusion(const HloInstruction* fusion_instr) {
  int64_t num_fusions = 0;
  if (fusion_instr->IsOutputFusion() || fusion_instr->IsLoopFusion()) {
    for (auto* instr :
         fusion_instr->fused_instructions_computation()->instructions()) {
      if (instr->IsOutputFusion() || instr->IsLoopFusion()) {
        auto cur_count = GetFusionCountInNestedFusion(instr);
        if (cur_count > num_fusions) {
          num_fusions = cur_count;
        }
      }
    }
  }
  if (fusion_instr->IsLoopFusion() || fusion_instr->IsOutputFusion()) {
    num_fusions += 1;
  }
  return num_fusions;
}

bool ModuleContainsLargeKeyValueSort(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kSort &&
          instruction->operand_count() > 1 &&
          instruction->operand(0)->shape().element_type() ==
              PrimitiveType::BF16 &&
          ShapeUtil::ElementsIn(instruction->operand(0)->shape()) >=
              (1 << 14)) {
        return true;
      }
    }
  }
  return false;
}

bool ModuleTestsFloatsForEquality(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCompare &&
          (instruction->comparison_direction() == ComparisonDirection::kEq ||
           instruction->comparison_direction() == ComparisonDirection::kNe) &&
          ShapeUtil::ElementIsFloating(instruction->operand(0)->shape())) {
        return true;
      }
    }
  }
  return false;
}

bool ComputationHasRng(const HloComputation* computation) {
  for (const HloInstruction* instruction :
       computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kRng) {
      return true;
    }
  }
  return false;
}

bool LiteralContainsInfOrNan(const LiteralSlice& literal) {
  if (literal.shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(literal.shape()); ++i) {
      if (LiteralContainsInfOrNan(LiteralSlice(literal, {i}))) {
        return true;
      }
    }
    return false;
  }
  bool contains_inf_or_nan = primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto type) -> bool {
        if constexpr (primitive_util::IsFloatingPointType(type)) {
          using NativeT = primitive_util::NativeTypeOf<type>;
          if (!std::numeric_limits<NativeT>::has_infinity &&
              !std::numeric_limits<NativeT>::has_quiet_NaN) {
            return false;
          }
          bool found = false;
          literal.EachCell<NativeT>(
              [&](absl::Span<const int64_t> /*indices*/, const NativeT& value) {
                if (std::isinf(value) || std::isnan(value)) {
                  found = true;
                }
              });
          return found;
        }
        return false;
      },
      literal.shape().element_type());
  return contains_inf_or_nan;
}

}  // namespace hlo_isolation
}  // namespace xla
