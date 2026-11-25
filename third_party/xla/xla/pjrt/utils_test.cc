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

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "benchmark/benchmark.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/array2d.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
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

Shape MakeUnlayoutF32Shape(int64_t d0, int64_t d1) {
  Shape s = ShapeUtil::MakeShape(F32, {d0, d1});
  LayoutUtil::ClearLayout(&s);
  return s;
}

absl::StatusOr<XlaComputation> BuildShardedIdentityComputation(
    const Shape& shape, const OpSharding& sharding) {
  XlaBuilder builder("sharded_identity");
  XlaOp p0 = Parameter(&builder, 0, shape, "p0");
  ABSL_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build(p0));
  for (HloInstructionProto& instr : *computation.mutable_proto()
                                         ->mutable_computations(0)
                                         ->mutable_instructions()) {
    *instr.mutable_sharding() = sharding;
  }
  return computation;
}

absl::StatusOr<Shape> ChooseDefaultLayout(Shape shape) {
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape;
}

TEST(UtilsTest, DetermineArgumentLayoutsNestedAndEmptyTuples) {
  Shape inner_tuple = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {6, 10}), ShapeUtil::MakeTupleShape({})});
  Shape param_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {8, 4}), inner_tuple,
                                 ShapeUtil::MakeTokenShape()});
  Shape out_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({}), ShapeUtil::MakeShape(F32, {12, 8})});

  OpSharding param_sharding =
      HloSharding::Tuple(param_shape, {HloSharding::IotaTile({2, 1}),
                                       HloSharding::IotaTile({1, 2}),
                                       HloSharding::Replicate()})
          .ToProto();
  OpSharding out_sharding =
      HloSharding::Tuple(out_shape, {HloSharding::IotaTile({3, 1})}).ToProto();

  XlaBuilder builder("nested_tuple");
  Parameter(&builder, 0, param_shape, "p0");
  XlaOp empty = Tuple(&builder, {});
  XlaOp c0 =
      ConstantR2FromArray2D<float>(&builder, Array2D<float>(12, 8, 1.0f));
  XlaOp root = Tuple(&builder, {empty, c0});
  ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build(root));

  HloComputationProto* entry =
      computation.mutable_proto()->mutable_computations(0);
  for (HloInstructionProto& instr : *entry->mutable_instructions()) {
    if (instr.opcode() == "parameter") {
      *instr.mutable_sharding() = param_sharding;
    } else if (instr.id() == entry->root_id()) {
      *instr.mutable_sharding() = out_sharding;
    }
  }

  std::vector<Shape> seen_sharded_shapes;
  auto record_and_choose_layout = [&](Shape shape) -> absl::StatusOr<Shape> {
    seen_sharded_shapes.push_back(shape);
    LayoutUtil::SetToDefaultLayout(&shape);
    return shape;
  };

  std::optional<std::vector<Shape>> arg_layouts;
  ExecutableBuildOptions build_options;
  std::vector<const Shape*> arg_layout_ptrs;
  EXPECT_THAT(DetermineArgumentLayoutsFromCompileOptions(
                  computation, record_and_choose_layout, arg_layouts,
                  &build_options, &arg_layout_ptrs),
              IsOk());

  EXPECT_THAT(seen_sharded_shapes, ElementsAre(MakeUnlayoutF32Shape(4, 4),
                                               MakeUnlayoutF32Shape(6, 5),
                                               MakeUnlayoutF32Shape(4, 8)));
}

TEST(UtilsTest, DetermineArgumentLayoutsReplicatedShardingOnTuple) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {8, 4}), ShapeUtil::MakeShape(F32, {6, 10})});
  ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                       BuildShardedIdentityComputation(
                           tuple_shape, HloSharding::Replicate().ToProto()));

  std::vector<Shape> seen_sharded_shapes;
  auto record_and_choose_layout = [&](Shape shape) -> absl::StatusOr<Shape> {
    seen_sharded_shapes.push_back(shape);
    LayoutUtil::SetToDefaultLayout(&shape);
    return shape;
  };

  std::optional<std::vector<Shape>> arg_layouts;
  ExecutableBuildOptions build_options;
  std::vector<const Shape*> arg_layout_ptrs;
  EXPECT_THAT(DetermineArgumentLayoutsFromCompileOptions(
                  computation, record_and_choose_layout, arg_layouts,
                  &build_options, &arg_layout_ptrs),
              IsOk());
  EXPECT_THAT(
      seen_sharded_shapes,
      ElementsAre(MakeUnlayoutF32Shape(8, 4), MakeUnlayoutF32Shape(6, 10),
                  MakeUnlayoutF32Shape(8, 4), MakeUnlayoutF32Shape(6, 10)));
}

TEST(UtilsTest, DetermineArgumentLayoutsTiledNonTupleShardingOnTupleFails) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {8, 4}), ShapeUtil::MakeShape(F32, {6, 10})});
  ASSERT_OK_AND_ASSIGN(
      XlaComputation computation,
      BuildShardedIdentityComputation(tuple_shape,
                                      HloSharding::IotaTile({2, 1}).ToProto()));

  std::optional<std::vector<Shape>> arg_layouts;
  ExecutableBuildOptions build_options;
  std::vector<const Shape*> arg_layout_ptrs;
  EXPECT_THAT(DetermineArgumentLayoutsFromCompileOptions(
                  computation, ChooseDefaultLayout, arg_layouts, &build_options,
                  &arg_layout_ptrs),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected tuple or non-tiled sharding")));
}

TEST(UtilsTest, DetermineArgumentLayoutsShortTupleShardingFails) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {8, 4}), ShapeUtil::MakeShape(F32, {6, 10})});
  OpSharding short_tuple;
  short_tuple.set_type(OpSharding::TUPLE);
  *short_tuple.add_tuple_shardings() = HloSharding::Replicate().ToProto();
  ASSERT_OK_AND_ASSIGN(
      XlaComputation computation,
      BuildShardedIdentityComputation(tuple_shape, short_tuple));

  std::optional<std::vector<Shape>> arg_layouts;
  ExecutableBuildOptions build_options;
  std::vector<const Shape*> arg_layout_ptrs;
  EXPECT_THAT(DetermineArgumentLayoutsFromCompileOptions(
                  computation, ChooseDefaultLayout, arg_layouts, &build_options,
                  &arg_layout_ptrs),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("leaf_index")));
}

TEST(UtilsTest, DetermineArgumentLayoutsOversizedTupleShardingFails) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {8, 4}), ShapeUtil::MakeShape(F32, {6, 10})});
  OpSharding oversized_tuple;
  oversized_tuple.set_type(OpSharding::TUPLE);
  for (int i = 0; i < 3; ++i) {
    *oversized_tuple.add_tuple_shardings() = HloSharding::Replicate().ToProto();
  }
  ASSERT_OK_AND_ASSIGN(
      XlaComputation computation,
      BuildShardedIdentityComputation(tuple_shape, oversized_tuple));

  std::optional<std::vector<Shape>> arg_layouts;
  ExecutableBuildOptions build_options;
  std::vector<const Shape*> arg_layout_ptrs;
  EXPECT_THAT(DetermineArgumentLayoutsFromCompileOptions(
                  computation, ChooseDefaultLayout, arg_layouts, &build_options,
                  &arg_layout_ptrs),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Tuple sharding has 3 elements")));
}

TEST(UtilsTest, DetermineArgumentLayoutsEmptyTupleSingleElementSharding) {
  Shape empty_tuple = ShapeUtil::MakeTupleShape({});
  OpSharding single_elem_tuple;
  single_elem_tuple.set_type(OpSharding::TUPLE);
  *single_elem_tuple.add_tuple_shardings() = HloSharding::Replicate().ToProto();
  ASSERT_OK_AND_ASSIGN(
      XlaComputation computation,
      BuildShardedIdentityComputation(empty_tuple, single_elem_tuple));

  std::optional<std::vector<Shape>> arg_layouts;
  ExecutableBuildOptions build_options;
  std::vector<const Shape*> arg_layout_ptrs;
  EXPECT_THAT(DetermineArgumentLayoutsFromCompileOptions(
                  computation, ChooseDefaultLayout, arg_layouts, &build_options,
                  &arg_layout_ptrs),
              IsOk());
}

void BM_DetermineArgumentLayoutsLargeTuple(benchmark::State& state) {
  const int num_leaves = state.range(0);
  std::vector<Shape> leaf_shapes;
  std::vector<HloSharding> leaf_shardings;
  leaf_shapes.reserve(num_leaves);
  leaf_shardings.reserve(num_leaves);
  for (int i = 0; i < num_leaves; ++i) {
    leaf_shapes.push_back(ShapeUtil::MakeShape(F32, {16, 8}));
    leaf_shardings.push_back(HloSharding::IotaTile({2, 1}));
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(leaf_shapes);
  OpSharding tuple_sharding =
      HloSharding::Tuple(tuple_shape, leaf_shardings).ToProto();

  absl::StatusOr<XlaComputation> computation =
      BuildShardedIdentityComputation(tuple_shape, tuple_sharding);
  CHECK_OK(computation);

  for (auto s : state) {
    std::optional<std::vector<Shape>> arg_layouts;
    ExecutableBuildOptions build_options;
    std::vector<const Shape*> arg_layout_ptrs;
    CHECK_OK(DetermineArgumentLayoutsFromCompileOptions(
        *computation, ChooseDefaultLayout, arg_layouts, &build_options,
        &arg_layout_ptrs));
    benchmark::DoNotOptimize(arg_layouts);
    benchmark::DoNotOptimize(arg_layout_ptrs);
  }
}

BENCHMARK(BM_DetermineArgumentLayoutsLargeTuple)
    ->Arg(1000)
    ->Arg(4000)
    ->Arg(16000)
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace xla
