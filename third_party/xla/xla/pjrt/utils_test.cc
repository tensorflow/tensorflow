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

#include "xla/pjrt/utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(UtilsTest, LayoutModeToXlaShape_AutoHostMemorySpace) {
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {8});
  Shape sharded_shape = unsharded_shape;

  // Auto layout mode with host memory space should fallback to default layout.
  LayoutMode layout_mode;
  layout_mode.mode = LayoutMode::Mode::kAuto;

  auto choose_compact_layout = [](Shape shape) -> absl::StatusOr<Shape> {
    *shape.mutable_layout() = LayoutUtil::MakeLayout({0});
    return shape;
  };

  TF_ASSERT_OK_AND_ASSIGN(
      Shape result,
      LayoutModeToXlaShape(layout_mode, unsharded_shape, sharded_shape,
                           Layout::kHostMemorySpace, choose_compact_layout));

  // The result should have a layout set, and the memory space should be host.
  EXPECT_TRUE(result.has_layout());
  EXPECT_EQ(result.layout().memory_space(), Layout::kHostMemorySpace);
}

TEST(UtilsTest, LayoutModeToXlaShape_AutoDeviceMemorySpace) {
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {8});
  Shape sharded_shape = unsharded_shape;

  // Auto layout mode with device memory space should leave layout and memory
  // space unset.
  LayoutMode layout_mode;
  layout_mode.mode = LayoutMode::Mode::kAuto;

  auto choose_compact_layout = [](Shape shape) -> absl::StatusOr<Shape> {
    *shape.mutable_layout() = LayoutUtil::MakeLayout({0});
    return shape;
  };

  TF_ASSERT_OK_AND_ASSIGN(
      Shape result,
      LayoutModeToXlaShape(layout_mode, unsharded_shape, sharded_shape,
                           Layout::kDefaultMemorySpace, choose_compact_layout));

  // The result should NOT have a layout set (because it's AUTO and not host).
  EXPECT_FALSE(result.has_layout());
}

TEST(UtilsTest, SwapMemorySpaceRejectedByMlirMemoryKinds) {
  EXPECT_EQ(SwapMemorySpace::kKind, "swap");
  EXPECT_NE(SwapMemorySpace::kKindId, PinnedHostMemorySpace::kKindId);
  EXPECT_NE(SwapMemorySpace::kKindId, UnpinnedHostMemorySpace::kKindId);

  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> arg_mod,
      ParseMlirModuleString("module { func.func @main(%arg0: tensor<4xi32> "
                            "{mhlo.memory_kind = \"swap\"}) -> tensor<4xi32> { "
                            "return %arg0 : tensor<4xi32> } }",
                            context));
  EXPECT_THAT(
      GetArgMemoryKinds(*arg_mod),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be used for compilation or execution")));

  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> out_mod,
      ParseMlirModuleString("module { func.func @main(%arg0: tensor<4xi32>) -> "
                            "(tensor<4xi32> {mhlo.memory_kind = \"swap\"}) { "
                            "return %arg0 : tensor<4xi32> } }",
                            context));
  EXPECT_THAT(
      GetOutputMemoryKinds(*out_mod),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be used for compilation or execution")));
}

}  // namespace
}  // namespace xla
