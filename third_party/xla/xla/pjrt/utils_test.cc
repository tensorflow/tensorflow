/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(LayoutPropertiesUtils, MlirLayoutIsSuccessfullyParsed) {
  constexpr char kProgram[] =
      R"(
      func.func @main(
        %arg0: tensor<1x1xf32> { mhlo.layout_mode = "default" },
        %arg1: tensor<1x1xf32> { mhlo.layout_mode = "{0,1}",
                                 mhlo.memory_kind = "unpinned_host" },
        %arg2: tensor<1x1xf32>,
        %arg3: tensor<1x1xf32> { mhlo.layout_mode = "auto" },
        %arg4: tensor<1x1xf32>
    ) -> (tensor<1x1xf32> { mhlo.layout_mode = "{0,1}" },
          tensor<1x1xf32>,
          tensor<1x1xf32> { mhlo.layout_mode = "auto" },
          tensor<1x1xf32> { mhlo.layout_mode = "default",
                            mhlo.memory_kind = "pinned_host" },
          tensor<1x1xf32>) {
      return %arg0, %arg1, %arg2, %arg3, %arg4 :
        tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
        tensor<1x1xf32>, tensor<1x1xf32>
    }
  )";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseMlirModuleString(kProgram, context));

  TF_ASSERT_OK_AND_ASSIGN(auto layout_properties,
                          GetModuleLayoutProperties(*module));

  EXPECT_THAT(
      layout_properties.arg_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{
              LayoutMode{LayoutMode::Mode::kUserSpecified, Layout({0, 1})},
              xla::Layout::kHostMemorySpace},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kAuto}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.arg_layout_properties.ToString();

  EXPECT_THAT(
      layout_properties.out_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{
              LayoutMode{LayoutMode::Mode::kUserSpecified, Layout({0, 1})}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kAuto}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault},
                                   xla::Layout::kHostMemorySpace},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.out_layout_properties.ToString();
}

TEST(LayoutPropertiesUtils, MlirTupledLayoutIsSuccessfullyParsed) {
  constexpr char kProgram[] =
      R"(
      func.func @main(%arg0: tuple<tensor<1x1xf32>, tensor<1x1xf32>>
    ) -> tuple<tensor<1x1xf32>, tensor<1x1xf32>> {
      return %arg0 : tuple<tensor<1x1xf32>, tensor<1x1xf32>>
    }
  )";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseMlirModuleString(kProgram, context));

  TF_ASSERT_OK_AND_ASSIGN(auto layout_properties,
                          GetModuleLayoutProperties(*module));

  EXPECT_THAT(
      layout_properties.arg_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.arg_layout_properties.ToString();

  EXPECT_THAT(
      layout_properties.out_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.out_layout_properties.ToString();
}

TEST(LayoutPropertiesUtils,
     HLOLayoutIsSuccessfullyParsedFromFrontendAttributes) {
  constexpr char kProgram[] = R"(
      HloModule module, frontend_attributes={
        arg_layout_modes="default;{0,1};default;auto",
        arg_memory_spaces="0;4",
        out_layout_modes="auto",
        out_memory_spaces="2"}
      
      ENTRY main {
        arg0 = f32[1,1] parameter(0)
        arg1 = f32[1,1] parameter(1)
        arg2 = f32[1,1] parameter(2)
        arg3 = f32[1,1] parameter(3)
        ROOT arg4 = f32[1,1] parameter(4)
      })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module,
      ParseAndReturnUnverifiedModule(
          kProgram, {}, HloParserOptions().set_fill_missing_layouts(false)));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(
      auto layout_properties,
      GetModuleLayoutPropertiesFromFrontendAttributes(xla_computation));

  EXPECT_THAT(
      layout_properties.arg_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{
              LayoutMode{LayoutMode::Mode::kUserSpecified, Layout({0, 1})}, 4},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kAuto}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.arg_layout_properties.ToString();

  EXPECT_THAT(layout_properties.out_layout_properties.items,
              ElementsAre(ArgumentLayoutProperties{
                  LayoutMode{LayoutMode::Mode::kAuto}, 2}))
      << layout_properties.out_layout_properties.ToString();
}

TEST(LayoutPropertiesUtils,
     HLOTupledLayoutIsSuccessfullyParsedFromFrontendAttributes) {
  constexpr char kProgram[] = R"(
      HloModule module, frontend_attributes={
        arg_layout_modes="default;{0,1};default;auto",
        arg_memory_spaces="0;4",
        out_layout_modes="auto;default;{0,1}",
        out_memory_spaces="2;0;3"}
      
      ENTRY main {
        ROOT arg0 = (f32[1,1], f32[1,1], f32[1,1], f32[1,1], f32[1,1]) parameter(0)
      })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module,
      ParseAndReturnUnverifiedModule(
          kProgram, {}, HloParserOptions().set_fill_missing_layouts(false)));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(
      auto layout_properties,
      GetModuleLayoutPropertiesFromFrontendAttributes(xla_computation));

  EXPECT_THAT(
      layout_properties.arg_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{
              LayoutMode{LayoutMode::Mode::kUserSpecified, Layout({0, 1})}, 4},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kAuto}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.arg_layout_properties.ToString();

  EXPECT_THAT(
      layout_properties.out_layout_properties.items,
      ElementsAre(
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kAuto}, 2},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{
              LayoutMode{LayoutMode::Mode::kUserSpecified, Layout({0, 1})}, 3},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0},
          ArgumentLayoutProperties{LayoutMode{LayoutMode::Mode::kDefault}, 0}))
      << layout_properties.out_layout_properties.ToString();
}

}  // namespace
}  // namespace xla
