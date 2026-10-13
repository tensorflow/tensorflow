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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace pjrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

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

std::vector<xla::PjRtPartialProgramProto> PrepareInputPartialPrograms(
    const std::string& next_phase, absl::string_view program_format) {
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
        xla::CpuTopology(
            std::vector<xla::CpuTopology::CpuDevice>(),
            xla::cpu::TargetMachineOptions(xla::GetDebugOptionsFromFlags())));
  }

  xla::MaybeOwningMlirModule ParseStablehloModule(
      absl::string_view module_str = kStablehloModuleStr) {
    auto context = std::make_shared<mlir::MLIRContext>();
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> stablehlo_module =
        mlir::stablehlo::parseStablehloModule(std::string(module_str),
                                              *context);
    CHECK(mlir::succeeded(stablehlo_module));
    return xla::MaybeOwningMlirModule(std::move(context),
                                      std::move(*stablehlo_module));
  }
};

// Test that the sample phase compiler's RegisterAllPhases method fails when
// attempting to register the same phase twice.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerRegisterAllPhases) {
  EXPECT_THAT(phase_compiler_->RegisterAllPhases(),
              absl_testing::StatusIs(absl::StatusCode::kAlreadyExists));
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
  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

// Test that the sample phase compiler's Compile method is not implemented for
// mlir::ModuleOp.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerCompileWithMlirModule) {
  xla::CompileOptions options;
  mlir::ModuleOp module;
  xla::PjRtClient* client = nullptr;
  auto status =
      phase_compiler_->Compile(options, xla::MaybeOwningMlirModule(module),
                               *topology_description_, client);
  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

// Test the correct usage of the RunPhases method of the sample phase compiler.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerRunPhases) {
  // Prepare the input programs.
  auto partial_programs_in = PrepareInputPartialPrograms(
      /*next_phase=*/std::string(phase_compile_sample_plugin::kPhaseName),
      /*program_format=*/kStablehloBytecodeFormat);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {
      std::string(phase_compile_sample_plugin::kPhaseName)};
  auto partial_programs_out = phase_compiler_->RunPhases(
      xla::CompileOptions(), std::move(partial_programs_in),
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
          /*program_format=*/kStablehloBytecodeFormat);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {};
  auto expected_programs = partial_programs_in;
  auto partial_programs_out = phase_compiler_->RunPhases(
      xla::CompileOptions(), std::move(partial_programs_in),
      *topology_description_, phases_to_run);

  TF_ASSERT_OK(partial_programs_out);

  for (size_t i = 0; i < expected_programs.size(); ++i) {
    EXPECT_THAT(partial_programs_out->at(i),
                EqualsProto(expected_programs.at(i)));
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
          /*program_format=*/kStablehloBytecodeFormat);

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"unregistered_phase_name"};
  auto partial_programs_out = phase_compiler_->RunPhases(
      xla::CompileOptions(), std::move(partial_programs_in),
      *topology_description_, phases_to_run);
  EXPECT_THAT(partial_programs_out,
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
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
  auto partial_programs_out = phase_compiler_->RunPhases(
      xla::CompileOptions(), std::move(partial_programs_in),
      *topology_description_, phases_to_run);
  EXPECT_THAT(partial_programs_out,
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input partial programs cannot be empty")));
}

// Plugin-specific validation: Test the RunPhases method of the sample phase
// compiler with unexpected input program format.
TEST_F(SamplePhaseCompilerTest, PluginSpecificValidationWithUnexpectedFormat) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgramProto> partial_programs_in =
      PrepareInputPartialPrograms(
          /*next_phase=*/std::string(phase_compile_sample_plugin::kPhaseName),
          /*program_format=*/"unexpected_format");

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {
      std::string(phase_compile_sample_plugin::kPhaseName)};
  auto partial_programs_out = phase_compiler_->RunPhases(
      xla::CompileOptions(), std::move(partial_programs_in),
      *topology_description_, phases_to_run);
  EXPECT_THAT(partial_programs_out,
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input programs are not in expected format")));
}

// Test the correct usage of the GetPhaseNames method of the sample phase
// compiler.
TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerGetPhaseNames) {
  auto phase_names_status = phase_compiler_->GetPhaseNames();
  EXPECT_THAT(phase_names_status,
              absl_testing::IsOkAndHolds(
                  ElementsAre(phase_compile_sample_plugin::kPhaseName)));
}

TEST_F(SamplePhaseCompilerTest, TestSamplePhaseCompilerCompileToRelocatable) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> relocatable,
      phase_compiler_->CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(), *topology_description_,
          /*name=*/"my_relocatable",
          /*is_entrypoint=*/false, /*client=*/nullptr));

  EXPECT_EQ(relocatable->name(), "my_relocatable");
  const auto* sample_relocatable =
      absl::down_cast<const phase_compile_sample_plugin::SampleRelocatable*>(
          relocatable.get());
  ASSERT_THAT(sample_relocatable->partial_programs(), SizeIs(1));
  EXPECT_EQ(sample_relocatable->partial_programs()[0].program_name(),
            "my_relocatable");
  EXPECT_EQ(sample_relocatable->partial_programs()[0].producer_phase(),
            phase_compile_sample_plugin::kPhaseName);

  mlir::MLIRContext context;
  EXPECT_THAT(
      phase_compile_sample_plugin::StablehloTypeSerialization::Deserialize(
          sample_relocatable->partial_programs()[0].program(), context),
      absl_testing::IsOk());
}

TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerCompileToRelocatableWithNullModule) {
  mlir::ModuleOp null_module;
  EXPECT_THAT(
      phase_compiler_->CompileToRelocatable(
          xla::CompileOptions(), xla::MaybeOwningMlirModule(null_module),
          *topology_description_, /*name=*/"bad",
          /*is_entrypoint=*/false, /*client=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerSerializeAndDeserializeRelocatable) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> relocatable,
      phase_compiler_->CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(), *topology_description_,
          /*name=*/"base",
          /*is_entrypoint=*/false, /*client=*/nullptr));

  ASSERT_OK_AND_ASSIGN(std::string serialized, relocatable->Serialize());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtRelocatable> deserialized,
                       phase_compiler_->DeserializeRelocatable(serialized));

  EXPECT_EQ(deserialized->name(), relocatable->name());
  EXPECT_EQ(deserialized->build_id(), relocatable->build_id());
  EXPECT_EQ(deserialized->call_backend_config(),
            relocatable->call_backend_config());

  const auto* sample_relocatable =
      absl::down_cast<const phase_compile_sample_plugin::SampleRelocatable*>(
          relocatable.get());
  const auto* sample_deserialized =
      absl::down_cast<const phase_compile_sample_plugin::SampleRelocatable*>(
          deserialized.get());
  ASSERT_EQ(sample_deserialized->partial_programs().size(),
            sample_relocatable->partial_programs().size());
  for (size_t i = 0; i < sample_relocatable->partial_programs().size(); ++i) {
    EXPECT_THAT(sample_deserialized->partial_programs()[i],
                EqualsProto(sample_relocatable->partial_programs()[i]));
  }
}

TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerDeserializeRelocatableWithInvalidInput) {
  EXPECT_THAT(phase_compiler_->DeserializeRelocatable(""),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(phase_compiler_->DeserializeRelocatable("not_a_valid_proto"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SamplePhaseCompilerTest,
       TestSamplePhaseCompilerLinkRelocatablesUnimplemented) {
  // We don't have a sample for LinkRelocatables since we don't have an easy
  // way to create a sample PjRtExecutable that actually works.
  EXPECT_THAT(phase_compiler_->LinkRelocatables(xla::CompileOptions(),
                                                *topology_description_,
                                                /*entrypoint=*/nullptr,
                                                /*dependencies=*/{}),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace pjrt
