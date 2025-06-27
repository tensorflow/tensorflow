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

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Api.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace pjrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::StatusIs;

constexpr absl::string_view kStablehloModuleStr = R"(
  module {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      %0 = stablehlo.constant dense<0> : tensor<4xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<4xi32>
      func.return %1 : tensor<4xi32>
    }
  }
  )";

std::vector<xla::PjRtPartialProgramProto> PrepareInputPartialPrograms(
    const std::string& next_phase, size_t program_format) {
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
  partial_program.set_generating_phase("n/a");
  partial_program.add_next_phases({next_phase});
  partial_program.set_version("1.0");

  return {partial_program};
}

class SamplePhaseCompilerTest : public ::testing::Test {
 protected:
  std::unique_ptr<pjrt::phase_compile_sample_plugin::SamplePhaseCompiler>
      phase_compiler_;
  std::unique_ptr<xla::PjRtTopologyDescription> topology_description_;

  SamplePhaseCompilerTest() {
    phase_compiler_ = std::make_unique<
        pjrt::phase_compile_sample_plugin::SamplePhaseCompiler>();
    CHECK_OK(phase_compiler_->RegisterAllPhases());

    topology_description_ = std::make_unique<xla::CpuTopologyDescription>(
        xla::CpuId(), xla::CpuName(), "<unknown>",
        std::vector<xla::CpuTopology::CpuDevice>(), std::vector<std::string>());
  }
};

// Test that the sample phase compiler's RegisterAllPhases method fails when
// attempting to register the same phase twice.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerRegisterAllPhases) {
  EXPECT_THAT(phase_compiler_->RegisterAllPhases(),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

// Test that the sample phase compiler's Compile method is not implemented for
// XlaComputation.
TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerCompileWithXlaComputation) {
  xla::CompileOptions options;
  xla::XlaComputation computation;
  xla::PjRtClient* client = nullptr;
  auto status = phase_compiler_->Compile(options, computation,
                                         *topology_description_, client);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnimplemented));
}

// Test that the sample phase compiler's Compile method is not implemented for
// mlir::ModuleOp.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerCompileWithMlirModule) {
  xla::CompileOptions options;
  mlir::ModuleOp module;
  xla::PjRtClient* client = nullptr;
  auto status =
      phase_compiler_->Compile(options, module, *topology_description_, client);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnimplemented));
}

// Test the correct usage of the RunPhases method of the sample phase compiler.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerRunPhases) {
  // Prepare the input programs.
  auto partial_programs_in = PrepareInputPartialPrograms(
      /*next_phase=*/std::string(phase_compile_sample_plugin::kPhaseName),
      /*program_format=*/0);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {
      std::string(phase_compile_sample_plugin::kPhaseName)};
  auto partial_programs_out =
      phase_compiler_->RunPhases(xla::CompileOptions(), partial_programs_in,
                                 *topology_description_, phases_to_run);

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

// Test that the RunPhases method of the sample phase compiler with empty phases
// to run will return the input programs as is.
TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerRunPhasesWithEmptyPhasesToRun) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(
          /*next_phase=*/std::string(phase_compile_sample_plugin::kPhaseName),
          /*program_format=*/0);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {};
  auto partial_programs_out =
      phase_compiler_->RunPhases(xla::CompileOptions(), partial_programs_in,
                                 *topology_description_, phases_to_run);

  TF_ASSERT_OK(partial_programs_out);

  for (size_t i = 0; i < partial_programs_in.size(); ++i) {
    EXPECT_THAT(partial_programs_out->at(i),
                EqualsProto(partial_programs_in.at(i)));
  }
}

// Test the RunPhases method of the sample phase compiler with an unregistered
// phase name.
TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerRunPhasesWithUnregisteredPhase) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(
          /*next_phase=*/std::string(phase_compile_sample_plugin::kPhaseName),
          /*program_format=*/0);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"unregistered_phase_name"};
  auto partial_programs_out =
      phase_compiler_->RunPhases(xla::CompileOptions(), partial_programs_in,
                                 *topology_description_, phases_to_run);
  EXPECT_THAT(partial_programs_out, StatusIs(absl::StatusCode::kNotFound));
}

// Plugin-specific validation: Test the RunPhases method of the sample phase
// compiler with empty input programs.
TEST_F(SamplePhaseCompilerTest,
       PluginSpecificValidationWithEmptyInputPrograms) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in = {};

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {
      std::string(phase_compile_sample_plugin::kPhaseName)};
  auto partial_programs_out =
      phase_compiler_->RunPhases(xla::CompileOptions(), partial_programs_in,
                                 *topology_description_, phases_to_run);
  EXPECT_THAT(partial_programs_out,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input partial programs cannot be empty")));
}

// Plugin-specific validation: Test the RunPhases method of the sample phase
// compiler with unexpected input program format.
TEST_F(SamplePhaseCompilerTest, PluginSpecificValidationWithUnexpectedFormat) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(
          /*next_phase=*/std::string(phase_compile_sample_plugin::kPhaseName),
          /*program_format=*/1  // 1 expresses some format which is not expected
                                // by the sample plugin for the kPhaseName
                                // phase.
      );

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {
      std::string(phase_compile_sample_plugin::kPhaseName)};
  auto partial_programs_out =
      phase_compiler_->RunPhases(xla::CompileOptions(), partial_programs_in,
                                 *topology_description_, phases_to_run);
  EXPECT_THAT(partial_programs_out,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input programs are not in expected format")));
}

// Test the correct usage of the GetPhaseNames method of the sample phase
// compiler.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerGetPhaseNames) {
  auto phase_names_status = phase_compiler_->GetPhaseNames();
  TF_EXPECT_OK(phase_names_status);
  std::vector<std::string> phase_names = *phase_names_status;
  EXPECT_THAT(phase_names,
              ElementsAre(phase_compile_sample_plugin::kPhaseName));
}

}  // namespace
}  // namespace pjrt
