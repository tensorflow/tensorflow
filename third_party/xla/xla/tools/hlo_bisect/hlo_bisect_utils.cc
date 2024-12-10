/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/tools/hlo_bisect/hlo_bisect_utils.h"

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/protobuf_util.h"
#include "xla/service/dump.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/prepare_reference_module.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/util.h"
#include "tsl/platform/path.h"

namespace xla {
namespace bisect {
namespace {

// Executes the module on the given platform with the given input and returns
// the result literal.
Literal ExecuteWithRunnerAndRetrieveResult(std::unique_ptr<HloModule> module,
                                           absl::Span<const Literal> input_data,
                                           HloRunnerInterface* runner,
                                           bool run_hlo_passes) {
  auto result_status =
      runner->Execute(std::move(module), input_data, run_hlo_passes);
  TF_CHECK_OK(result_status.status())
      << "Failed to execute on " << runner->Name();
  return std::move(result_status).value();
}

// Loads the given HloProto as HloModule.
absl::StatusOr<std::unique_ptr<HloModule>> LoadModuleFromHloProto(
    const HloProto& proto) {
  const HloModuleProto& module_proto = proto.hlo_module();
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, GetDebugOptionsFromFlags()));
  return CreateModuleFromProto(module_proto, module_config);
}

absl::StatusOr<std::unique_ptr<HloModule>>
LoadModuleAndInputDataFromHloSnapshot(const HloSnapshot& snapshot,
                                      std::vector<Literal>* input_data) {
  for (int64_t i = 0; i < snapshot.arguments_size(); ++i) {
    TF_ASSIGN_OR_RETURN(Literal literal,
                        Literal::CreateFromProto(snapshot.arguments(i)));
    input_data->push_back(std::move(literal));
  }
  TF_ASSIGN_OR_RETURN(
      HloModuleConfig config,
      HloModule::CreateModuleConfigFromProto(snapshot.hlo().hlo_module(),
                                             xla::GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(snapshot.hlo().hlo_module(), config);
}

absl::StatusOr<ModuleWithInputs> GetModuleAndInputData(
    absl::string_view input_filename) {
  const std::string input_file(input_filename);
  tsl::Env* env = tsl::Env::Default();
  std::unique_ptr<HloModule> module;

  HloSnapshot hlo_snapshot;
  if (tsl::ReadBinaryProto(env, input_file, &hlo_snapshot).ok()) {
    std::vector<Literal> input_data;
    TF_ASSIGN_OR_RETURN(module, LoadModuleAndInputDataFromHloSnapshot(
                                    hlo_snapshot, &input_data));
    CHECK_EQ(module->entry_computation()->num_parameters(), input_data.size());
    return std::make_pair(std::move(module), std::move(input_data));
  }
  LOG(INFO) << input_file << " is not HloSnapshot. Trying HLO binary proto.\n";
  HloProto hlo_proto;
  absl::StatusOr<std::unique_ptr<HloModule>> module_or_status;
  if (tsl::ReadBinaryProto(env, input_file, &hlo_proto).ok()) {
    module_or_status = LoadModuleFromHloProto(hlo_proto);
    if (!module_or_status.ok()) {
      LOG(ERROR) << "Failed to load hlo proto"
                 << module_or_status.status().message();
      return module_or_status.status();
    }
    module = std::move(module_or_status).value();
    return std::make_pair(std::move(module), std::vector<Literal>());
  }
  LOG(INFO) << input_file << " is not HloProto. Trying HLO text.\n";
  std::string hlo_string;
  absl::Status to_string_status =
      tsl::ReadFileToString(env, input_file, &hlo_string);
  if (!to_string_status.ok()) {
    LOG(ERROR) << input_file << " problem in reading file to string: "
               << to_string_status.message();
    return to_string_status;
  }

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  module_or_status = ParseAndReturnUnverifiedModule(hlo_string, config);
  if (!module_or_status.ok()) {
    LOG(ERROR) << input_file << " is not HLO text either, error in parsing "
               << module_or_status.status().message();
    return module_or_status.status();
  }

  module = std::move(module_or_status).value();
  return std::make_pair(std::move(module), std::vector<Literal>());
}

// Outputs the given HloModule as HloProto to the given file.
absl::Status DumpHloModule(HloModule* module, const std::string& file_name,
                           absl::string_view dir_path,
                           absl::string_view output_format) {
  HloProto proto = MakeHloProto(*module);
  if (output_format == "hlo") {
    tsl::Env* env = tsl::Env::Default();
    TF_RETURN_IF_ERROR(CreateDirIfNeeded(std::string(dir_path), env));
    std::string file_path =
        tsl::io::JoinPath(dir_path, SanitizeFileName(file_name)) + ".hlo";
    LOG(INFO) << "Dumped HLO text to " << file_path;
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
        env, file_path,
        module->ToString(HloPrintOptions::Canonical()
                             .set_print_large_constants(true)
                             .set_compact_operands(false))));
  } else if (output_format == "pb") {
    std::string path;
    TF_RETURN_IF_ERROR(
        DumpProtoToDirectory(proto, std::string(dir_path), file_name, &path));
    LOG(INFO) << "Dumped HLO module proto to " << path;

  } else {
    LOG(FATAL) << "Unexpected output format: " << output_format;
  }

  return absl::OkStatus();
}

}  // namespace

MiscompareChecker::MiscompareChecker(HloModule* module,
                                     std::vector<Literal>&& input_data,
                                     absl::string_view test_platform,
                                     absl::string_view reference_platform,
                                     ErrorSpec error_spec)
    : error_spec_(error_spec) {
  // Generate input data and store the data for all the execution.
  std::minstd_rand0 rng_engine;
  if (input_data.empty()) {
    absl::StatusOr<std::vector<Literal>> input_status =
        MakeFakeArguments(module, &rng_engine);
    CHECK(input_status.ok());
    input_data_ = std::move(input_status).value();
  } else {
    VLOG(2) << "Using provided input data";
    input_data_ = std::move(input_data);
  }

  // Set up the reference platform.
  absl::StatusOr<se::Platform*> reference_platform_status =
      PlatformUtil::GetPlatform(std::string(reference_platform));
  CHECK(reference_platform_status.ok());
  reference_runner_ =
      std::make_unique<HloRunner>(reference_platform_status.value());

  // Set up the test platform.
  absl::StatusOr<se::Platform*> test_platform_status =
      PlatformUtil::GetPlatform(std::string(test_platform));
  CHECK(test_platform_status.ok());
  test_runner_ =
      std::make_unique<HloRunner>(std::move(test_platform_status).value());
}

// Executes the module with the test_runner and the reference_runner and
// compares the results from the two runs. Returns true if the two results are
// not near to indicate a bug exists.
absl::StatusOr<bool> MiscompareChecker::Run(const HloModule& module) {
  std::unique_ptr<HloModule> test_module = module.Clone(/*suffix=*/"");

  // Make sure that the module config has a non-zero seed, which the CPU and GPU
  // backends will use for kRng random number generation. A zero seed in the
  // module config, on the other hand, tells the backend to generate new seeds
  // for kRng random number generation, which can lead to different results from
  // the two backends.
  if (test_module->config().seed() == 0) {
    HloModuleConfig config = test_module->config();
    config.set_seed(42);
    test_module->set_config(config);
  }

  // Prepare the reference module.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> reference_module,
                      PrepareReferenceModule(*test_module, test_runner_.get()));

  // Run the module on the reference platform.
  Literal reference_result = ExecuteWithRunnerAndRetrieveResult(
      std::move(reference_module), input_data_, reference_runner_.get(),
      /*run_hlo_passes=*/true);

  // Run the module on the test platform.
  Literal test_result = ExecuteWithRunnerAndRetrieveResult(
      std::move(test_module), input_data_, test_runner_.get(),
      /*run_hlo_passes=*/true);

  // Compare the results.
  absl::StatusOr<::testing::AssertionResult> status_or_result =
      LiteralTestUtil::Near(/*expected=*/reference_result,
                            /*actual=*/test_result,
                            /*error_spec=*/error_spec_,
                            /*detailed_message=*/true);

  CHECK(status_or_result.ok())
      << "Problem with running the clone module, may be there is a problem in "
         "the cloned module itself?";
  return !static_cast<bool>(std::move(status_or_result).value());
}

absl::flat_hash_map<std::string, Literal> MiscompareChecker::GetResults() {
  return {};
}

absl::StatusOr<std::unique_ptr<HloModule>>
MiscompareChecker::PrepareReferenceModule(
    const HloModule& hlo_module, HloRunnerInterface* hlo_runner) const {
  // By default clone the test module (could be overridden).
  return xla::PrepareReferenceModule(hlo_module, hlo_runner);
}

absl::StatusOr<bool> ScriptChecker::Run(const HloModule& module) {
  tsl::Env* env = tsl::Env::Default();
  // Write hlo into a temporary file.
  std::string hlo_path;
  if (!env->LocalTempFilename(&hlo_path)) {
    return Internal("couldn't get temp HLO file name");
  }

  absl::Cleanup hlo_cleaner = [&] {
    TF_CHECK_OK(tsl::Env::Default()->DeleteFile(hlo_path));
  };

  std::string hlo_contents =
      module.ToString(HloPrintOptions::Canonical()
                          .set_print_large_constants(true)
                          .set_compact_operands(false));

  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, hlo_path, hlo_contents));

  tsl::SubProcess script_subprocess;
  std::vector<std::string> script_args = {path_to_script_, hlo_path};

  script_subprocess.SetProgram(path_to_script_, script_args);
  script_subprocess.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  script_subprocess.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!script_subprocess.Start()) {
    return Internal("Failed to launch script");
  }

  std::string stderr_output;
  std::string stdout_output;
  int exit_status = script_subprocess.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/&stdout_output,
      /*stderr_output=*/&stderr_output);
  VLOG(3) << "Standard output";
  VLOG(3) << stdout_output;
  VLOG(3) << "Standard error output";
  VLOG(3) << stderr_output;
  VLOG(3) << "Exit status from " << path_to_script_ << " " << exit_status;
  return /*has_bug=*/exit_status != 0;
}

absl::flat_hash_map<std::string, Literal> ScriptChecker::GetResults() {
  return {};
}

absl::StatusOr<std::unique_ptr<HloModule>> BisectRunner::RunEntry() {
  HloBisectState hlo_bisect(std::move(module_), bug_checker_.get());
  TF_ASSIGN_OR_RETURN(bool has_bug, hlo_bisect.ShouldProcess());
  if (!has_bug) {
    return InvalidArgument(
        "Don't see the bug in the unmodified module. Something is not right. "
        "Can't bisect.");
  }

  TF_RETURN_IF_ERROR(hlo_bisect.TrimEntryComputation().status());
  return hlo_bisect.GetResult();
}

absl::StatusOr<std::unique_ptr<HloModule>> BisectRunner::RunAll() {
  std::unique_ptr<HloModule> original_module = std::move(module_);
  std::unique_ptr<HloModule> result;
  for (HloComputation* c : original_module->computations()) {
    LOG(INFO) << "Bisecting computation: " << c->name();
    module_ = original_module->Clone(/*suffix=*/"");
    absl::StatusOr<std::unique_ptr<HloModule>> new_result;
    if (c->IsEntryComputation()) {
      // Run on the entry computation with input data.
      new_result = RunEntry();
    } else {
      // Run on a non-entry computation with no input data (use random).
      HloComputation* new_entry = module_->GetComputationWithName(c->name());
      CHECK(new_entry != nullptr) << "Missing computation: " << c->name();
      module_->ReplaceEntryComputation(new_entry);
      new_result = RunEntry();
      if (new_result.status().code() ==
          tensorflow::error::Code::INVALID_ARGUMENT) {
        VLOG(2) << "The bug is unaffected by the computation " << c->name();
        continue;
      }
    }
    if (!new_result.ok()) {
      return new_result;
    }
    if (result == nullptr ||
        result->computation_count() > new_result.value()->computation_count()) {
      result = std::move(new_result.value());
    }
  }
  return result;
}

void RunBisect(std::unique_ptr<BisectRunner> runner, bool all_computations,
               absl::string_view dump_path, absl::string_view output_format) {
  absl::StatusOr<std::unique_ptr<HloModule>> bisect_status =
      all_computations ? runner->RunAll() : runner->RunEntry();
  CHECK(bisect_status.ok()) << bisect_status.status().message();

  std::unique_ptr<HloModule> new_module = std::move(bisect_status.value());
  absl::Status dump_status =
      DumpHloModule(new_module.get(), new_module->name() + "_trimmed",
                    dump_path, output_format);
  CHECK(dump_status.ok()) << dump_status.message();
}

absl::StatusOr<ModuleWithInputs> GetVerifiedModuleAndInputData(
    absl::string_view input_filename) {
  std::unique_ptr<HloModule> module;
  std::vector<Literal> input_data;
  TF_ASSIGN_OR_RETURN(std::tie(module, input_data),
                      GetModuleAndInputData(input_filename));

  // If any instruction doesn't have a layout, set to default layout.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!LayoutUtil::HasLayout(instruction->shape())) {
        LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
      }
    }
  }
  absl::Status verified_status = HloVerifier(/*layout_sensitive=*/false,
                                             /*allow_mixed_precision=*/false)
                                     .Run(module.get())
                                     .status();
  if (!verified_status.ok()) {
    LOG(ERROR) << "Failed to verify hlo module " << verified_status.message();
    return verified_status;
  }

  return std::make_pair(std::move(module), std::move(input_data));
}

}  // namespace bisect
}  // namespace xla
