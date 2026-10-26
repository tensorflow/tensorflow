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
#include "xla/pjrt/pjrt_relocatable.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/layout_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::SizeIs;

class PjRtRelocatableTest : public ::testing::Test {
 protected:
  PjRtRelocatableTest() {
    phase_compiler_ = std::make_unique<
        pjrt::phase_compile_sample_plugin::SamplePhaseCompiler>();
    CHECK_OK(phase_compiler_->RegisterAllPhases());

    topology_description_ = std::make_unique<CpuTopologyDescription>(
        CpuId(), CpuName(), "<unknown>",
        CpuTopology(std::vector<CpuTopology::CpuDevice>(),
                    cpu::TargetMachineOptions(GetDebugOptionsFromFlags())));
  }

  absl::StatusOr<std::unique_ptr<PjRtRelocatable>> CompileToRelocatable(
      absl::string_view module_str, absl::string_view name = "test_relocatable",
      bool is_entrypoint = false) {
    auto context = std::make_shared<mlir::MLIRContext>();
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> stablehlo_module =
        mlir::stablehlo::parseStablehloModule(std::string(module_str),
                                              *context);
    CHECK(mlir::succeeded(stablehlo_module));
    return phase_compiler_->CompileToRelocatable(
        CompileOptions(),
        MaybeOwningMlirModule(std::move(context), std::move(*stablehlo_module)),
        *topology_description_, name, is_entrypoint,
        /*client=*/nullptr);
  }

  std::unique_ptr<pjrt::phase_compile_sample_plugin::SamplePhaseCompiler>
      phase_compiler_;
  std::unique_ptr<PjRtTopologyDescription> topology_description_;
};

TEST_F(PjRtRelocatableTest, BasicMetadataAndSingleOutput) {
  constexpr absl::string_view kModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<2xi32>) -> tensor<4x8xf32> {
        %0 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
        %1 = stablehlo.add %arg0, %0 : tensor<4x8xf32>
        func.return %1 : tensor<4x8xf32>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtRelocatable> relocatable,
                       CompileToRelocatable(kModuleStr, "add_op"));

  EXPECT_EQ(relocatable->name(), "add_op");
  EXPECT_THAT(relocatable->build_id(), Not(IsEmpty()));
  EXPECT_THAT(relocatable->call_backend_config(), Not(IsEmpty()));
  EXPECT_FALSE(relocatable->has_side_effects());

  EXPECT_THAT(relocatable->GetParameterElementTypes(),
              absl_testing::IsOkAndHolds(ElementsAre(F32, S32)));
  EXPECT_THAT(relocatable->GetOutputElementTypes(),
              absl_testing::IsOkAndHolds(ElementsAre(F32)));

  EXPECT_THAT(relocatable->GetParameterDimensions(),
              absl_testing::IsOkAndHolds(
                  ElementsAre(DimensionVector{4, 8}, DimensionVector{2})));
  EXPECT_THAT(relocatable->GetOutputDimensions(),
              absl_testing::IsOkAndHolds(ElementsAre(DimensionVector{4, 8})));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> param_layouts,
      relocatable->GetParameterLayouts());
  ASSERT_THAT(param_layouts, SizeIs(2));
  EXPECT_EQ(*param_layouts[0], PjRtLayout(LayoutUtil::MakeDescendingLayout(2)));
  EXPECT_EQ(*param_layouts[1], PjRtLayout(LayoutUtil::MakeDescendingLayout(1)));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> output_layouts,
      relocatable->GetOutputLayouts());
  ASSERT_THAT(output_layouts, SizeIs(1));
  EXPECT_EQ(*output_layouts[0],
            PjRtLayout(LayoutUtil::MakeDescendingLayout(2)));

  EXPECT_THAT(relocatable->GetParameterOutputAliases(),
              absl_testing::IsOkAndHolds(IsEmpty()));
}

TEST_F(PjRtRelocatableTest, MultipleOutputsAndAliases) {
  constexpr absl::string_view kModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 1 : i32},
                      %arg1: tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<4xf32>) {
        func.return %arg1, %arg0 : tensor<2x3xi64>, tensor<4xf32>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtRelocatable> relocatable,
                       CompileToRelocatable(kModuleStr, "swap_op"));

  EXPECT_THAT(relocatable->GetParameterElementTypes(),
              absl_testing::IsOkAndHolds(ElementsAre(F32, S64)));
  EXPECT_THAT(relocatable->GetOutputElementTypes(),
              absl_testing::IsOkAndHolds(ElementsAre(S64, F32)));

  EXPECT_THAT(relocatable->GetParameterDimensions(),
              absl_testing::IsOkAndHolds(
                  ElementsAre(DimensionVector{4}, DimensionVector{2, 3})));
  EXPECT_THAT(relocatable->GetOutputDimensions(),
              absl_testing::IsOkAndHolds(
                  ElementsAre(DimensionVector{2, 3}, DimensionVector{4})));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> output_layouts,
      relocatable->GetOutputLayouts());
  ASSERT_THAT(output_layouts, SizeIs(2));
  EXPECT_EQ(*output_layouts[0],
            PjRtLayout(LayoutUtil::MakeDescendingLayout(2)));
  EXPECT_EQ(*output_layouts[1],
            PjRtLayout(LayoutUtil::MakeDescendingLayout(1)));

  EXPECT_THAT(relocatable->GetParameterOutputAliases(),
              absl_testing::IsOkAndHolds(ElementsAre(Pair(0, 1))));
}

TEST_F(PjRtRelocatableTest, HasSideEffects) {
  constexpr absl::string_view kSideEffectModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
        %0 = stablehlo.custom_call @side_effecting_op(%arg0) {has_side_effect = true} : (tensor<4xi32>) -> tensor<4xi32>
        func.return %0 : tensor<4xi32>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtRelocatable> relocatable,
      CompileToRelocatable(kSideEffectModuleStr, "side_effect_op"));
  EXPECT_TRUE(relocatable->has_side_effects());
}

TEST_F(PjRtRelocatableTest, SerializeAndDeserializeRoundTrip) {
  constexpr absl::string_view kModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<4x2xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4x2xf32> {
        %0 = stablehlo.custom_call @side_effecting_op(%arg0) {has_side_effect = true} : (tensor<4x2xf32>) -> tensor<4x2xf32>
        func.return %0 : tensor<4x2xf32>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtRelocatable> original,
                       CompileToRelocatable(kModuleStr, "round_trip_op"));
  ASSERT_OK_AND_ASSIGN(std::string serialized, original->Serialize());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtRelocatable> deserialized,
                       phase_compiler_->DeserializeRelocatable(serialized));

  EXPECT_EQ(deserialized->name(), original->name());
  EXPECT_EQ(deserialized->build_id(), original->build_id());
  EXPECT_EQ(deserialized->call_backend_config(),
            original->call_backend_config());
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
      std::vector<std::shared_ptr<const PjRtLayout>> orig_param_layouts,
      original->GetParameterLayouts());
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> deser_param_layouts,
      deserialized->GetParameterLayouts());
  ASSERT_EQ(deser_param_layouts.size(), orig_param_layouts.size());
  for (size_t i = 0; i < orig_param_layouts.size(); ++i) {
    EXPECT_EQ(*deser_param_layouts[i], *orig_param_layouts[i]);
  }

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> orig_out_layouts,
      original->GetOutputLayouts());
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> deser_out_layouts,
      deserialized->GetOutputLayouts());
  ASSERT_EQ(deser_out_layouts.size(), orig_out_layouts.size());
  for (size_t i = 0; i < orig_out_layouts.size(); ++i) {
    EXPECT_EQ(*deser_out_layouts[i], *orig_out_layouts[i]);
  }
}

}  // namespace
}  // namespace xla
