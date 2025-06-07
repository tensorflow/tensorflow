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
#include "xla/pjrt/c/pjrt_c_api_phase_compile_cpu.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
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
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_partial_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_partial_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_cpu_internal.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_partial_program.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"

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

std::vector<xla::PjRtPartialProgram> PrepareInputPartialPrograms() {
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

  xla::PjRtPartialProgram partial_program;
  partial_program.SetProgram(bytecode_status.value());
  partial_program.SetFormat(0);  // 0 expresses StableHLO bytecode per plugin
  partial_program.SetGeneratingPhase("n/a");
  partial_program.SetNextPhases({"stablehlo_to_optimized_stablehlo"});
  partial_program.SetVersion("1.0");
  return {partial_program};
}

const PJRT_PhaseCompile_Extension* GetPhaseCompileExtension(
    const PJRT_Api* api) {
  return pjrt::FindExtension<PJRT_PhaseCompile_Extension>(
      api, PJRT_Extension_Type::PJRT_Extension_Type_PhaseCompile);
}

}  // namespace

class PhaseCompileTest : public ::testing::Test {
 protected:
  static const PJRT_PhaseCompile_Extension* partial_compile_extension_;
  static xla::PjRtTopologyDescription* topology_description_;

  static void SetUpTestSuite() {
    partial_compile_extension_ = GetPhaseCompileExtension(GetPjrtApi());

    std::vector<std::string> machine_attributes;
    machine_attributes.push_back("abc");
    xla::CpuTopologyDescription* cpu_topology_description_impl =
        new xla::CpuTopologyDescription(xla::CpuId(), xla::CpuName(),
                                        "<unknown>",
                                        /*cpu_devices=*/{}, machine_attributes);
    topology_description_ = cpu_topology_description_impl;
  }

  static void TearDownTestSuite() { delete topology_description_; }
};

const PJRT_PhaseCompile_Extension*
    PhaseCompileTest::partial_compile_extension_ = nullptr;
xla::PjRtTopologyDescription* PhaseCompileTest::topology_description_ = nullptr;

TEST_F(PhaseCompileTest, LookupExtension) {
  ASSERT_NE(partial_compile_extension_, nullptr);
}

TEST_F(PhaseCompileTest, RunPhases) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgram> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = pjrt::RunPhases(
      partial_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_OK(partial_programs_out);

  // Print the output programs.
  for (auto& partial_program : partial_programs_out.value()) {
    mlir::MLIRContext context;
    auto deserialized_module =
        phase_compile_cpu_plugin::StableHLOTypeSerialization::deserialize(
            partial_program.GetProgram(), context);
    EXPECT_OK(deserialized_module);
    deserialized_module.value()->dump();
  }

  // Destroy the partial programs.
  for (auto& partial_program : partial_programs_in) {
    partial_program.Destroy();
  }
  for (auto& partial_program : partial_programs_out.value()) {
    partial_program.Destroy();
  }
}

// Running a phase with empty input programs should fail.
TEST_F(PhaseCompileTest, ConsumeEmptyPrograms) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgram> partial_programs_in = {};

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {std::string(kPhaseName)};
  auto partial_programs_out = pjrt::RunPhases(
      partial_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Input partial programs cannot be empty");

  // Destroy the partial programs.
  for (auto& partial_program : partial_programs_in) {
    partial_program.Destroy();
  }
}

// Running a phase with empty "phases to run" should fail.
TEST_F(PhaseCompileTest, ConsumeEmptyPhases) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgram> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {};
  auto partial_programs_out = pjrt::RunPhases(
      partial_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Phases to run cannot be empty");

  // Destroy the partial programs.
  for (auto& partial_program : partial_programs_in) {
    partial_program.Destroy();
  }
}

// Running a phase with empty phase name should fail.
TEST_F(PhaseCompileTest, ConsumeEmptyPhaseName) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgram> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {""};
  auto partial_programs_out = pjrt::RunPhases(
      partial_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Phase name cannot be empty");

  // Destroy the partial programs.
  for (auto& partial_program : partial_programs_in) {
    partial_program.Destroy();
  }
}

// Running a phase which is not expected to be run on the input programs should
// fail.
TEST_F(PhaseCompileTest, ConsumeProgramWithIncompatiblePhase) {
  // Prepare the input programs.
  std::vector<xla::PjRtPartialProgram> partial_programs_in =
      PrepareInputPartialPrograms();

  // Run the partial compile phase.
  std::vector<std::string> phases_to_run = {"IllegalPhaseName"};
  auto partial_programs_out = pjrt::RunPhases(
      partial_compile_extension_, xla::CompileOptions(), *topology_description_,
      partial_programs_in, phases_to_run);
  EXPECT_FALSE(partial_programs_out.ok());
  EXPECT_EQ(partial_programs_out.status().message(),
            "Input partial program cannot be consumed by a phase with name "
            "IllegalPhaseName");

  // Destroy the partial programs.
  for (auto& partial_program : partial_programs_in) {
    partial_program.Destroy();
  }
}

TEST_F(PhaseCompileTest, GetPhaseNames) {
  std::vector<std::string> phase_names =
      pjrt::GetPhaseNames(partial_compile_extension_).value();
  EXPECT_EQ(phase_names.size(), 1) << "Failure: Incorrect number of phases.";
  EXPECT_EQ(phase_names[0], kPhaseName);
}

}  // namespace pjrt
