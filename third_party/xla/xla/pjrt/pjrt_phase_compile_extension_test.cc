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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/reference/Api.h"
#include "xla/layout_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_phase_compiler.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/pjrt/string_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace pjrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::SizeIs;

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
  EXPECT_NE(phase_compile_extension.relocatable_destroy, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_name, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_build_id, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_call_backend_config, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_has_side_effects, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_serialize, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_parameter_element_types,
            nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_output_element_types, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_parameter_dimensions, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_output_dimensions, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_parameter_layouts, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_output_layouts, nullptr);
  EXPECT_NE(phase_compile_extension.relocatable_parameter_output_aliases,
            nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_compile_to_relocatable,
            nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_deserialize_relocatable,
            nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_link_relocatables, nullptr);
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
      xla::CompileOptions(), std::move(partial_programs_in),
      *topology_description_, phases_to_run);
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
      xla::CompileOptions(), std::move(partial_programs_in),
      *topology_description_, phases_to_run);
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

TEST_F(PhaseCompileExtensionTest, CompileToRelocatableAndMetadata) {
  constexpr absl::string_view kModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<4x8xf32> {tf.aliasing_output = 1 : i32},
                      %arg1: tensor<2xi32>) -> (tensor<2xi32>, tensor<4x8xf32>) {
        %0 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
        %1 = stablehlo.add %arg0, %0 : tensor<4x8xf32>
        func.return %arg1, %1 : tensor<2xi32>, tensor<4x8xf32>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> relocatable,
      phase_compile_extension_wrapper_->CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(kModuleStr),
          *topology_description_, /*name=*/"my_relocatable",
          /*is_entrypoint=*/false, client_.get()));

  EXPECT_EQ(relocatable->name(), "my_relocatable");
  EXPECT_THAT(relocatable->build_id(), Not(IsEmpty()));
  EXPECT_THAT(relocatable->call_backend_config(), Not(IsEmpty()));
  EXPECT_FALSE(relocatable->has_side_effects());

  EXPECT_THAT(relocatable->GetParameterElementTypes(),
              absl_testing::IsOkAndHolds(ElementsAre(xla::F32, xla::S32)));
  EXPECT_THAT(relocatable->GetOutputElementTypes(),
              absl_testing::IsOkAndHolds(ElementsAre(xla::S32, xla::F32)));

  EXPECT_THAT(relocatable->GetParameterDimensions(),
              absl_testing::IsOkAndHolds(ElementsAre(xla::DimensionVector{4, 8},
                                                     xla::DimensionVector{2})));
  EXPECT_THAT(relocatable->GetOutputDimensions(),
              absl_testing::IsOkAndHolds(ElementsAre(
                  xla::DimensionVector{2}, xla::DimensionVector{4, 8})));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> param_layouts,
      relocatable->GetParameterLayouts());
  ASSERT_THAT(param_layouts, SizeIs(2));
  EXPECT_EQ(*param_layouts[0],
            xla::PjRtLayout(xla::LayoutUtil::MakeDescendingLayout(2)));
  EXPECT_EQ(*param_layouts[1],
            xla::PjRtLayout(xla::LayoutUtil::MakeDescendingLayout(1)));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts,
      relocatable->GetOutputLayouts());
  ASSERT_THAT(output_layouts, SizeIs(2));
  EXPECT_EQ(*output_layouts[0],
            xla::PjRtLayout(xla::LayoutUtil::MakeDescendingLayout(1)));
  EXPECT_EQ(*output_layouts[1],
            xla::PjRtLayout(xla::LayoutUtil::MakeDescendingLayout(2)));

  EXPECT_THAT(relocatable->GetParameterOutputAliases(),
              absl_testing::IsOkAndHolds(ElementsAre(Pair(0, 1))));
}

TEST_F(PhaseCompileExtensionTest, CompileToRelocatableWithNullModule) {
  mlir::ModuleOp null_module;
  EXPECT_THAT(
      phase_compile_extension_wrapper_->CompileToRelocatable(
          xla::CompileOptions(), xla::MaybeOwningMlirModule(null_module),
          *topology_description_, /*name=*/"bad",
          /*is_entrypoint=*/false, client_.get()),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PhaseCompileExtensionTest,
       CompileToRelocatableWithoutClientUsesStableHloVersion) {
  const PJRT_Api* c_api =
      pjrt::phase_compile_sample_plugin::GetSamplePhaseCompilePjrtApi();
  ASSERT_NE(c_api, nullptr);
  PJRT_PhaseCompile_Extension custom_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();
  custom_extension.phase_compile_compile_to_relocatable =
      [](PJRT_PhaseCompile_CompileToRelocatable_Args* args) -> PJRT_Error* {
    EXPECT_EQ(args->client, nullptr);
    mlir::vhlo::Version version = mlir::vhlo::Version::getCurrentVersion();
    std::string version_string = absl::StrFormat(
        "%d.%d.%d", version.getMajor(), version.getMinor(), version.getPatch());
    EXPECT_TRUE(llvm::StringRef(args->program->code, args->program->code_size)
                    .contains(version_string));
    return pjrt::phase_compile_sample_plugin::
        CreateSamplePhaseCompileExtension()
            .phase_compile_compile_to_relocatable(args);
  };

  PJRT_PhaseCompile_Get_Compiler_Args get_compiler_args{};
  get_compiler_args.struct_size =
      PJRT_PhaseCompile_Get_Compiler_Args_STRUCT_SIZE;
  get_compiler_args.extension_start = nullptr;
  PJRT_Error* error =
      custom_extension.phase_compile_get_compiler(&get_compiler_args);
  ASSERT_EQ(error, nullptr);

  xla::PjRtCApiPhaseCompiler phase_compiler(c_api, &custom_extension,
                                            get_compiler_args.phase_compiler);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> relocatable,
      phase_compiler.CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(), *topology_description_,
          /*name=*/"aot_relocatable", /*is_entrypoint=*/false,
          /*client=*/nullptr));
  EXPECT_EQ(relocatable->name(), "aot_relocatable");
}

TEST_F(PhaseCompileExtensionTest, SerializeAndDeserializeRelocatable) {
  constexpr absl::string_view kModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<4x2xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4x2xf32> {
        %0 = stablehlo.custom_call @side_effecting_op(%arg0) {has_side_effect = true} : (tensor<4x2xf32>) -> tensor<4x2xf32>
        func.return %0 : tensor<4x2xf32>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> original,
      phase_compile_extension_wrapper_->CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(kModuleStr),
          *topology_description_, /*name=*/"round_trip_op",
          /*is_entrypoint=*/false, client_.get()));

  ASSERT_OK_AND_ASSIGN(std::string serialized, original->Serialize());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> deserialized,
      phase_compile_extension_wrapper_->DeserializeRelocatable(serialized));

  EXPECT_EQ(deserialized->name(), original->name());
  EXPECT_EQ(deserialized->build_id(), original->build_id());
  EXPECT_EQ(deserialized->call_backend_config(),
            original->call_backend_config());
  EXPECT_TRUE(deserialized->has_side_effects());
  EXPECT_EQ(deserialized->has_side_effects(), original->has_side_effects());
  EXPECT_EQ(deserialized->GetParameterElementTypes(),
            original->GetParameterElementTypes());
  EXPECT_EQ(deserialized->GetOutputElementTypes(),
            original->GetOutputElementTypes());
  EXPECT_EQ(deserialized->GetParameterDimensions(),
            original->GetParameterDimensions());
  EXPECT_EQ(deserialized->GetOutputDimensions(),
            original->GetOutputDimensions());
  EXPECT_EQ(deserialized->GetParameterOutputAliases(),
            original->GetParameterOutputAliases());

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> orig_param_layouts,
      original->GetParameterLayouts());
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> deser_param_layouts,
      deserialized->GetParameterLayouts());
  ASSERT_EQ(deser_param_layouts.size(), orig_param_layouts.size());
  for (size_t i = 0; i < orig_param_layouts.size(); ++i) {
    EXPECT_EQ(*deser_param_layouts[i], *orig_param_layouts[i]);
  }

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> orig_out_layouts,
      original->GetOutputLayouts());
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> deser_out_layouts,
      deserialized->GetOutputLayouts());
  ASSERT_EQ(deser_out_layouts.size(), orig_out_layouts.size());
  for (size_t i = 0; i < orig_out_layouts.size(); ++i) {
    EXPECT_EQ(*deser_out_layouts[i], *orig_out_layouts[i]);
  }
}

TEST_F(PhaseCompileExtensionTest, DeserializeRelocatableWithInvalidInput) {
  EXPECT_THAT(phase_compile_extension_wrapper_->DeserializeRelocatable(""),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(phase_compile_extension_wrapper_->DeserializeRelocatable(
                  "not_a_valid_proto"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PhaseCompileExtensionTest, LinkRelocatables) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> entrypoint,
      phase_compile_extension_wrapper_->CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(), *topology_description_,
          /*name=*/"entrypoint",
          /*is_entrypoint=*/true, client_.get()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtRelocatable> dependency,
      phase_compile_extension_wrapper_->CompileToRelocatable(
          xla::CompileOptions(), ParseStablehloModule(), *topology_description_,
          /*name=*/"dependency",
          /*is_entrypoint=*/false, client_.get()));

  EXPECT_THAT(
      phase_compile_extension_wrapper_->LinkRelocatables(
          xla::CompileOptions(), *topology_description_,
          std::shared_ptr<const xla::PjRtRelocatable>(std::move(entrypoint)),
          {std::shared_ptr<const xla::PjRtRelocatable>(std::move(dependency))}),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace pjrt
