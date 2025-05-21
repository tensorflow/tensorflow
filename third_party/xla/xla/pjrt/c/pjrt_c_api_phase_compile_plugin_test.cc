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
#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin.h"

#include <cstdlib>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/reference/Api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin_internal.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace pjrt {
namespace {

constexpr absl::string_view kStablehloModuleStr = R"(
  module {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.constant dense<0> : tensor<4xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<4xi32>
      func.return %1 : tensor<4xi32>
    }
  }
  )";

constexpr absl::string_view kPhaseName = "stablehlo_to_optimized_stablehlo";
std::vector<xla::PjRtPartialProgramProto> PrepareInputPartialPrograms(
    const std::string& next_phase) {
  std::string program_code{kStablehloModuleStr};

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> stablehlo_module =
      mlir::stablehlo::parseStablehloModule(program_code, context).value();

  auto bytecode_status =
      pjrt::phase_compile_cpu_plugin::StableHLOTypeSerialization::serialize(
          stablehlo_module);
  if (!bytecode_status.ok()) {
    exit(1);
  }

  xla::PjRtPartialProgramProto partial_program;
  partial_program.set_program(bytecode_status.value());
  partial_program.set_program_format(
      0);  // 0 expresses StableHLO bytecode per plugin
  partial_program.set_generating_phase("n/a");
  partial_program.add_next_phases({next_phase});
  partial_program.set_version("1.0");
  return {partial_program};
}

}  // namespace

class PhaseCompileTest : public ::testing::Test {
 protected:
  static const PJRT_Api* api_;
  static const PJRT_Api* api_with_invalid_extension_start_;
  static const PJRT_Api* api_with_invalid_platform_;
  static void SetUpTestSuite() {
    api_ = GetPjrtApi(/*test_flag_null_phase_extension=*/false,
                      /*test_flag_illegal_platform=*/false);
    api_with_invalid_extension_start_ =
        GetPjrtApi(/*test_flag_null_phase_extension=*/true,
                   /*test_flag_illegal_platform=*/false);
    api_with_invalid_platform_ =
        GetPjrtApi(/*test_flag_null_phase_extension=*/false,
                   /*test_flag_illegal_platform=*/true);
  }
  static void TearDownTestSuite() {}
};

const PJRT_Api* PhaseCompileTest::api_ = nullptr;
const PJRT_Api* PhaseCompileTest::api_with_invalid_extension_start_ = nullptr;
const PJRT_Api* PhaseCompileTest::api_with_invalid_platform_ = nullptr;

/***********************************
// Testing the RunPhases API.
***********************************/

// Test the correct usage of the RunPhases API.
TEST_F(PhaseCompileTest, RunPhases) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(std::string(kPhaseName));

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = pjrt::RunPhases(
      api_, xla::CompileOptions(), partial_programs_in, phases_to_run);
  EXPECT_OK(partial_programs_out);

  // Print the output programs.
  for (auto& partial_program : partial_programs_out.value()) {
    mlir::MLIRContext context;
    auto deserialized_module =
        phase_compile_cpu_plugin::StableHLOTypeSerialization::deserialize(
            partial_program.program(), context);
    EXPECT_OK(deserialized_module);
  }
}

// Test the RunPhases API with empty input programs.
TEST_F(PhaseCompileTest, ConsumeEmptyPrograms) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in = {};

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = pjrt::RunPhases(
      api_, xla::CompileOptions(), partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Input partial programs cannot be empty");
}

// Test the RunPhases API with an empty phases to run.
TEST_F(PhaseCompileTest, ConsumeEmptyPhases) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(std::string(kPhaseName));

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {};
  auto partial_programs_out = pjrt::RunPhases(
      api_, xla::CompileOptions(), partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Phases to run cannot be empty");
}

// Test the RunPhases API with an empty phase name.
TEST_F(PhaseCompileTest, ConsumeEmptyPhaseName) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(std::string(kPhaseName));

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {""};
  auto partial_programs_out = pjrt::RunPhases(
      api_, xla::CompileOptions(), partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Phase name cannot be empty");
}

// Test the RunPhases API with an incompatible phase: one that is not expected
// to be run on the input programs.
TEST_F(PhaseCompileTest, ConsumeProgramWithIncompatiblePhase) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(std::string(kPhaseName));

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"IllegalPhaseName"};
  auto partial_programs_out = pjrt::RunPhases(
      api_, xla::CompileOptions(), partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Input partial program cannot be consumed by a phase with name "
            "IllegalPhaseName");
}

// Test the RunPhases API with unregistered phases.
TEST_F(PhaseCompileTest, UnregisteredPhase) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms("SomeOtherPhase");

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"SomeOtherPhase"};
  auto partial_programs_out = pjrt::RunPhases(
      api_, xla::CompileOptions(), partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(
      partial_programs_out.status().message(),
      "No phase compiler/validator registered with phase name SomeOtherPhase");
}

// Test the RunPhases API with an invalid extension start.
TEST_F(PhaseCompileTest, RunPhasesWithInvalidExtensionStart) {
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(std::string(kPhaseName));
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};

  auto phase_names_status =
      pjrt::RunPhases(api_with_invalid_extension_start_, xla::CompileOptions(),
                      partial_programs_in, phases_to_run);
  EXPECT_FALSE(phase_names_status.ok());
  EXPECT_EQ(phase_names_status.status().message(),
            "Phase compile extension not found");
}

// Test the GetPhaseNames API with an invalid platform name.
TEST_F(PhaseCompileTest, RunPhasesWithInvalidPlatform) {
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(std::string(kPhaseName));
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};

  auto phase_names_status =
      pjrt::RunPhases(api_with_invalid_platform_, xla::CompileOptions(),
                      partial_programs_in, phases_to_run);
  EXPECT_FALSE(phase_names_status.ok());
  EXPECT_EQ(phase_names_status.status().message(),
            "No pjrt phase compiler registered for platform cuda");
}

/***********************************
// Testing the GetPhaseNames API.
***********************************/

// Test the correct usage of the GetPhaseNames API.
TEST_F(PhaseCompileTest, GetPhaseNames) {
  auto phase_names_status = pjrt::GetPhaseNames(api_);
  EXPECT_OK(phase_names_status);
  std::vector<std::string> phase_names = phase_names_status.value();
  EXPECT_EQ(phase_names.size(), 1);
  EXPECT_EQ(phase_names[0], kPhaseName);
}

// Test the GetPhaseNames API with an invalid extension start.
TEST_F(PhaseCompileTest, GetPhaseNamesWithInvalidExtensionStart) {
  auto phase_names_status =
      pjrt::GetPhaseNames(api_with_invalid_extension_start_);
  EXPECT_FALSE(phase_names_status.ok());
  EXPECT_EQ(phase_names_status.status().message(),
            "Phase compile extension not found");
}

// Test the GetPhaseNames API with an invalid platform name.
TEST_F(PhaseCompileTest, GetPhaseNamesWithInvalidPlatform) {
  auto phase_names_status = pjrt::GetPhaseNames(api_with_invalid_platform_);
  EXPECT_FALSE(phase_names_status.ok());
  EXPECT_EQ(phase_names_status.status().message(),
            "No pjrt phase compiler registered for platform cuda");
}

}  // namespace pjrt
