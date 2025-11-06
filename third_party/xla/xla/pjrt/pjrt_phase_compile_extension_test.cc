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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/pjrt/string_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace pjrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr absl::string_view kStablehloModuleStr = R"(
  module {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.constant dense<0> : tensor<4xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<4xi32>
      func.return %1 : tensor<4xi32>
    }
  }
  )";

constexpr absl::string_view kStablehloBytecodeFormat = "bytecode";
constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";

// We don't need a plugin registration target for this test, but we need to
// ensure that the plugin is registered before the first use of the phase
// compiler. Thus, we use a static initializer to ensure that the plugin is
// registered before the PhaseCompileExtensionTest class is instantiated.
bool sample_plugin_registered = []() {
  auto status = pjrt::SetPjrtApi(
      "sample-cpu",
      pjrt::phase_compile_sample_plugin::GetSamplePhaseCompilePjrtApi());
  return status.ok();
}();

std::vector<xla::PjRtPartialProgramProto> PrepareInputPartialPrograms(
    const std::string& next_phase, const absl::string_view program_format) {
  std::string program_code{kStablehloModuleStr};

  mlir::MLIRContext context;
  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> stablehlo_module =
      mlir::stablehlo::parseStablehloModule(program_code, context);
  CHECK(mlir::succeeded(stablehlo_module));

  auto bytecode_status =
      pjrt::phase_compile_sample_plugin::StablehloTypeSerialization::Serialize(
          *stablehlo_module);
  CHECK_OK(bytecode_status);

  xla::PjRtPartialProgramProto partial_program;
  partial_program.set_program(*bytecode_status);
  partial_program.set_program_format(program_format);
  partial_program.set_producer_phase("n/a");
  partial_program.add_consumer_phases({next_phase});
  partial_program.set_version("1.0");

  return {partial_program};
}

class PhaseCompileExtensionTest : public ::testing::Test {
 protected:
  std::unique_ptr<xla::PjRtPhaseCompiler> phase_compile_extension_wrapper_;
  const xla::PjRtTopologyDescription* topology_description_;
  std::unique_ptr<xla::PjRtClient> client_;

  PhaseCompileExtensionTest() {
    auto phase_compile_extension = xla::GetCApiPhaseCompiler("sample-cpu");
    phase_compile_extension_wrapper_ = std::move(*phase_compile_extension);

    // Create a topology description.
    auto client_or_status = xla::GetCApiClient("sample-cpu");
    CHECK_OK(client_or_status);
    client_ = std::move(*client_or_status);
    auto topology_or_status = client_->GetTopologyDescription();
    CHECK_OK(topology_or_status);
    topology_description_ = *topology_or_status;
  }
};

// Test that the phase compile extension has none of the callbacks set
// to null.
TEST(BasicPhaseCompileExtensionTest, TestExtensionRegistration) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  EXPECT_NE(phase_compile_extension.phase_compile_get_compiler, nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_destroy_compiler, nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_run_phases, nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_get_phase_names, nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_c_buffers_destroy, nullptr);
}

// Test the correct usage of the (1) GetCompiler, (2) DestroyCompiler, and (3)
// CBuffersDestroy APIs.
TEST(BasicPhaseCompileExtensionTest,
     TestPhaseCompileExtensionForGetCompilerDestroyCompilerAndCBuffersDestroy) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  // Get the phase compiler.
  PJRT_PhaseCompile_Get_Compiler_Args get_compiler_args;
  get_compiler_args.struct_size = sizeof(PJRT_PhaseCompile_Get_Compiler_Args);
  get_compiler_args.extension_start = nullptr;
  PJRT_Error* error =
      phase_compile_extension.phase_compile_get_compiler(&get_compiler_args);
  ASSERT_EQ(error, nullptr);

  // Get the phases names.
  PJRT_PhaseCompile_Get_PhaseNames_Args get_phase_names_args;
  get_phase_names_args.struct_size =
      sizeof(PJRT_PhaseCompile_Get_PhaseNames_Args);
  get_phase_names_args.extension_start = nullptr;
  get_phase_names_args.phase_compiler = get_compiler_args.phase_compiler;
  error = phase_compile_extension.phase_compile_get_phase_names(
      &get_phase_names_args);
  ASSERT_EQ(error, nullptr);
  // Convert the C-style phase names to C++ strings.
  std::vector<std::string> converted_strings =
      xla::ConvertCharBuffersToCppStrings(
          absl::MakeSpan(get_phase_names_args.phase_names,
                         get_phase_names_args.num_phase_names),
          absl::MakeConstSpan(get_phase_names_args.phase_names_sizes,
                              get_phase_names_args.num_phase_names));

  // Destroy the C-style buffer.
  PJRT_PhaseCompile_C_Buffers_Destroy_Args destroy_c_buffers_args;
  destroy_c_buffers_args.struct_size =
      sizeof(PJRT_PhaseCompile_C_Buffers_Destroy_Args);
  destroy_c_buffers_args.extension_start = &phase_compile_extension.base;
  destroy_c_buffers_args.char_buffers = get_phase_names_args.phase_names;
  destroy_c_buffers_args.char_buffer_sizes =
      get_phase_names_args.phase_names_sizes;
  destroy_c_buffers_args.num_char_buffers =
      get_phase_names_args.num_phase_names;
  phase_compile_extension.phase_compile_c_buffers_destroy(
      &destroy_c_buffers_args);

  // Destroy the phase compiler.
  PJRT_PhaseCompile_Destroy_Compiler_Args destroy_compiler_args;
  destroy_compiler_args.struct_size =
      sizeof(PJRT_PhaseCompile_Destroy_Compiler_Args);
  destroy_compiler_args.extension_start = nullptr;
  destroy_compiler_args.phase_compiler = get_compiler_args.phase_compiler;
  phase_compile_extension.phase_compile_destroy_compiler(
      &destroy_compiler_args);
}

// Test the correct usage of the RunPhases API.
TEST_F(PhaseCompileExtensionTest, RunPhases) {
  // Prepare the input programs.
  auto partial_programs_in = PrepareInputPartialPrograms(
      /*next_phase=*/std::string(kPhaseName),
      /*program_format=*/kStablehloBytecodeFormat);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = phase_compile_extension_wrapper_->RunPhases(
      xla::CompileOptions(), partial_programs_in, *topology_description_,
      phases_to_run);
  TF_ASSERT_OK(partial_programs_out);

  // Verify that the output programs are deserializable.
  for (auto& partial_program : *partial_programs_out) {
    mlir::MLIRContext context;
    auto deserialized_module =
        phase_compile_sample_plugin::StablehloTypeSerialization::Deserialize(
            partial_program.program(), context);
    TF_EXPECT_OK(deserialized_module);
  }
}

// Plugin-agnostic validation: Test the RunPhases API with an empty phases to
// run.
TEST_F(PhaseCompileExtensionTest,
       PluginAgnosticValidationWithEmptyPhasesToRun) {
  // Prepare the input programs.
  auto partial_programs_in =
      PrepareInputPartialPrograms(/*next_phase=*/std::string(kPhaseName),
                                  /*program_format=*/kStablehloBytecodeFormat);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {};
  auto partial_programs_out = phase_compile_extension_wrapper_->RunPhases(
      xla::CompileOptions(), partial_programs_in, *topology_description_,
      phases_to_run);
  EXPECT_THAT(
      partial_programs_out,
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Phases to run cannot be empty")));
}

// Plugin-agnostic validation: Test the RunPhases API with an incompatible
// phase: one that is not expected to be run on the input programs.
TEST_F(PhaseCompileExtensionTest,
       PluginAgnosticValidationWithIncompatiblePhase) {
  // Prepare the input programs.
  auto partial_programs_in =
      PrepareInputPartialPrograms(/*next_phase=*/std::string(kPhaseName),
                                  /*program_format=*/kStablehloBytecodeFormat);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"IllegalPhaseName"};
  auto partial_programs_out = phase_compile_extension_wrapper_->RunPhases(
      xla::CompileOptions(), partial_programs_in, *topology_description_,
      phases_to_run);
  EXPECT_THAT(
      partial_programs_out,
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input partial programs cannot be compiled by a phase "
                    "with name")));
}

// Test the correct usage of the GetPhaseNames API.
TEST_F(PhaseCompileExtensionTest, TestPhaseCompileExtensionForGetPhaseNames) {
  auto phase_names_status = phase_compile_extension_wrapper_->GetPhaseNames();
  EXPECT_THAT(phase_names_status,
              absl_testing::IsOkAndHolds(ElementsAre(kPhaseName)));
}

}  // namespace
}  // namespace pjrt
