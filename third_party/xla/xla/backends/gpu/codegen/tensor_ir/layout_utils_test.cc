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

#include "xla/backends/gpu/codegen/tensor_ir/layout_utils.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Support/TCutegen.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::tensor_ir {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
namespace tcg = ::mlir::nv_tensor_ir::tcutegen;

TEST(LayoutUtilsTest, ComputeStrideStringRowMajor2D) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {1, 0});
  EXPECT_EQ(ComputeStrideString(shape), "(16,1)");

  auto stride = tcg::Stride::fromString(ComputeStrideString(shape));
  ASSERT_TRUE(stride.has_value());
  EXPECT_EQ(stride->toString(), "(16,1)");
}

TEST(LayoutUtilsTest, ComputeStrideStringColumnMajor2D) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {0, 1});
  EXPECT_EQ(ComputeStrideString(shape), "(1,4)");

  auto stride = tcg::Stride::fromString(ComputeStrideString(shape));
  ASSERT_TRUE(stride.has_value());
  EXPECT_EQ(stride->toString(), "(1,4)");
}

TEST(LayoutUtilsTest, ComputeStrideStringRank1) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {8}, {0});
  EXPECT_EQ(ComputeStrideString(shape), "(1)");

  auto stride = tcg::Stride::fromString(ComputeStrideString(shape));
  ASSERT_TRUE(stride.has_value());
  EXPECT_EQ(stride->toString(), "(1)");
}

TEST(LayoutUtilsTest, ComputeStrideStringRank0Scalar) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  EXPECT_EQ(ComputeStrideString(shape), "()");

  // Verify that "()" is parsed by tcg::Stride::fromString.
  auto stride = tcg::Stride::fromString(ComputeStrideString(shape));
  ASSERT_TRUE(stride.has_value());
  EXPECT_EQ(stride->toString(), "()");
}

TEST(LayoutUtilsTest, ComputeStrideStringTransposed3D) {
  // f32[2,3,4] with minor_to_major {0,2,1}
  // dim 0 is most minor: stride = 1, product = 2
  // dim 2 is next: stride = 2, product = 2 * 4 = 8
  // dim 1 is most major: stride = 8, product = 8 * 3 = 24
  // Strides in dim order (0, 1, 2): (1, 8, 2)
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {0, 2, 1});
  EXPECT_EQ(ComputeStrideString(shape), "(1,8,2)");

  auto stride = tcg::Stride::fromString(ComputeStrideString(shape));
  ASSERT_TRUE(stride.has_value());
  EXPECT_EQ(stride->toString(), "(1,8,2)");
}

TEST(LayoutUtilsTest, ComputeStrideStringDegenerateDimOfSize1) {
  // f32[1, 16] with minor_to_major {1, 0}: dim 1 stride = 1, dim 0 stride = 16
  // -> "(16,1)"
  Shape row_major = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 16}, {1, 0});
  EXPECT_EQ(ComputeStrideString(row_major), "(16,1)");

  // f32[1, 16] with minor_to_major {0, 1}: dim 0 stride = 1, dim 1 stride = 1
  // -> "(1,1)"
  Shape col_major = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 16}, {0, 1});
  EXPECT_EQ(ComputeStrideString(col_major), "(1,1)");

  auto stride = tcg::Stride::fromString(ComputeStrideString(col_major));
  ASSERT_TRUE(stride.has_value());
  EXPECT_EQ(stride->toString(), "(1,1)");
}

TEST(LayoutUtilsTest, HasDefaultLayoutTests) {
  EXPECT_TRUE(HasDefaultLayout(ShapeUtil::MakeShape(F32, {})));
  EXPECT_TRUE(
      HasDefaultLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {8}, {0})));
  EXPECT_TRUE(HasDefaultLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {1, 0})));
  EXPECT_FALSE(HasDefaultLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {0, 1})));
  EXPECT_TRUE(HasDefaultLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {2, 1, 0})));
  EXPECT_FALSE(HasDefaultLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {0, 2, 1})));
  EXPECT_FALSE(HasDefaultLayout(ShapeUtil::MakeTupleShape({})));

  Shape no_layout = ShapeUtil::MakeShape(F32, {4, 16});
  no_layout.clear_layout();
  EXPECT_TRUE(HasDefaultLayout(no_layout));
}

// Builds a bare `nv_tensor_ir.graph` with the given argument/result types. The
// body is left empty: these tests only look at boundary attributes.
mlir::nv_tensor_ir::GraphOp MakeGraph(mlir::OpBuilder& b,
                                      llvm::ArrayRef<mlir::Type> arg_types,
                                      llvm::ArrayRef<mlir::Type> result_types) {
  auto graph = mlir::nv_tensor_ir::GraphOp::create(
      b, mlir::UnknownLoc::get(b.getContext()), "test_graph",
      /*sym_visibility=*/nullptr, b.getFunctionType(arg_types, result_types),
      /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
  graph.addEntryBlock();
  return graph;
}

class AttachLayoutStridesTest : public HloHardwareIndependentTestBase {};

TEST_F(AttachLayoutStridesTest, DefaultAndNonDefaultLayouts) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    fused_computation {
      p0 = f32[4,16]{1,0} parameter(0)
      p1 = f32[4,16]{0,1} parameter(1)
      p2 = f32[] parameter(2)
      b2 = f32[4,16]{0,1} broadcast(p2), dimensions={}
      add0 = f32[4,16]{0,1} add(p0, p1)
      ROOT add = f32[4,16]{0,1} add(add0, b2)
    }

    ENTRY entry {
      p0 = f32[4,16]{1,0} parameter(0)
      p1 = f32[4,16]{0,1} parameter(1)
      p2 = f32[] parameter(2)
      ROOT fusion = f32[4,16]{0,1} fusion(p0, p1, p2), kind=kCustom, calls=fused_computation
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));
  const HloComputation& computation =
      *Cast<HloFusionInstruction>(
           hlo_module->entry_computation()->root_instruction())
           ->fused_instructions_computation();

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto scalar_t = mlir::RankedTensorType::get({}, b.getF32Type());
  auto graph = MakeGraph(b, {t, t, scalar_t}, {t});

  ASSERT_THAT(AttachLayoutStrides(graph, computation), IsOk());

  auto stride_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getStrideAttrName();

  // Arg 0 is row-major, i.e. the default layout -> no stride attribute.
  EXPECT_EQ(graph.getArgAttr(0, stride_attr_name), nullptr);

  // Arg 1 is column-major -> stride "(1,4)".
  auto a1_stride =
      graph.getArgAttrOfType<mlir::StringAttr>(1, stride_attr_name);
  ASSERT_TRUE(a1_stride != nullptr);
  EXPECT_EQ(a1_stride.getValue().str(), "(1,4)");

  // Arg 2 is a rank-0 scalar, which always counts as default -> no attribute.
  EXPECT_EQ(graph.getArgAttr(2, stride_attr_name), nullptr);

  // The root is column-major -> stride "(1,4)".
  auto res_stride =
      graph.getResultAttrOfType<mlir::StringAttr>(0, stride_attr_name);
  ASSERT_TRUE(res_stride != nullptr);
  EXPECT_EQ(res_stride.getValue().str(), "(1,4)");

  graph.erase();
}

// The whole point of splitting strides out of the emitter is that a plain
// computation, with no fusion instruction and no buffer assignment, is enough
// to produce them.
TEST_F(AttachLayoutStridesTest, WorksOnAComputationWithoutAFusion) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY entry {
      p0 = f32[8,16,32]{2,0,1} parameter(0)
      ROOT out = f32[16,8,32]{2,1,0} bitcast(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));
  const HloComputation& computation = *hlo_module->entry_computation();

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto in_t = mlir::RankedTensorType::get({8, 16, 32}, b.getF32Type());
  auto out_t = mlir::RankedTensorType::get({16, 8, 32}, b.getF32Type());
  auto graph = MakeGraph(b, {in_t}, {out_t});

  ASSERT_THAT(AttachLayoutStrides(graph, computation), IsOk());

  auto stride_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getStrideAttrName();
  auto a0_stride =
      graph.getArgAttrOfType<mlir::StringAttr>(0, stride_attr_name);
  ASSERT_TRUE(a0_stride != nullptr);
  EXPECT_EQ(a0_stride.getValue().str(), "(32,256,1)");
  // The root is row-major -> no stride attribute.
  EXPECT_EQ(graph.getResultAttr(0, stride_attr_name), nullptr);

  graph.erase();
}

TEST_F(AttachLayoutStridesTest, ErrorPathGraphNumArgumentsMismatch) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY entry {
      p0 = f32[4,16]{1,0} parameter(0)
      ROOT id = f32[4,16]{1,0} copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  // Graph has 2 arguments, but the computation has 1 parameter.
  auto t0 = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t0, t0}, {t0});

  EXPECT_THAT(AttachLayoutStrides(graph, *hlo_module->entry_computation()),
              StatusIs(absl::StatusCode::kInternal));

  graph.erase();
}

TEST_F(AttachLayoutStridesTest, ErrorPathTupleParameter) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY entry {
      p0 = (f32[4], f32[4]) parameter(0)
      gte0 = f32[4] get-tuple-element(p0), index=0
      ROOT id = f32[4] copy(gte0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t0 = mlir::RankedTensorType::get({4}, b.getF32Type());
  auto graph = MakeGraph(b, {t0}, {t0});

  EXPECT_THAT(AttachLayoutStrides(graph, *hlo_module->entry_computation()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  graph.erase();
}

TEST_F(AttachLayoutStridesTest, MultipleResultsFromRootTuple) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY entry {
      p0 = f32[4,16]{1,0} parameter(0)
      c0 = f32[4,16]{0,1} copy(p0)
      c1 = f32[4,16]{1,0} copy(p0)
      ROOT t = (f32[4,16]{0,1}, f32[4,16]{1,0}) tuple(c0, c1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t}, {t, t});

  ASSERT_THAT(AttachLayoutStrides(graph, *hlo_module->entry_computation()),
              IsOk());

  auto stride_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getStrideAttrName();
  auto res0 = graph.getResultAttrOfType<mlir::StringAttr>(0, stride_attr_name);
  ASSERT_TRUE(res0 != nullptr);
  EXPECT_EQ(res0.getValue().str(), "(1,4)");
  EXPECT_TRUE(graph.getResultAttrOfType<mlir::StringAttr>(
                  1, stride_attr_name) == nullptr);

  graph.erase();
}

TEST_F(AttachLayoutStridesTest, ErrorPathResultCountMismatch) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY entry {
      p0 = f32[4,16]{1,0} parameter(0)
      ROOT id = f32[4,16]{1,0} copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  // Graph has 2 results, but the computation has 1.
  auto t = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t}, {t, t});

  EXPECT_THAT(AttachLayoutStrides(graph, *hlo_module->entry_computation()),
              StatusIs(absl::StatusCode::kInternal));

  graph.erase();
}

class AttachBufferAlignmentsTest : public HloHardwareIndependentTestBase {};

TEST_F(AttachBufferAlignmentsTest, AttachesAlignmentToArgsAndResult) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t, t}, {t});

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {1, 0});
  std::vector<emitters::KernelArgument> kernel_args;
  kernel_args.reserve(3);
  for (int64_t alignment : {128, 64, 256}) {
    kernel_args.emplace_back(shape);
    kernel_args.back().set_alignment(alignment);
  }

  ASSERT_THAT(AttachBufferAlignments(graph, kernel_args), IsOk());

  auto alignment_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getAlignmentAttrName();
  auto a0 = graph.getArgAttrOfType<mlir::IntegerAttr>(0, alignment_attr_name);
  ASSERT_TRUE(a0 != nullptr);
  EXPECT_EQ(a0.getInt(), 128);
  auto a1 = graph.getArgAttrOfType<mlir::IntegerAttr>(1, alignment_attr_name);
  ASSERT_TRUE(a1 != nullptr);
  EXPECT_EQ(a1.getInt(), 64);
  auto res =
      graph.getResultAttrOfType<mlir::IntegerAttr>(0, alignment_attr_name);
  ASSERT_TRUE(res != nullptr);
  EXPECT_EQ(res.getInt(), 256);

  graph.erase();
}

TEST_F(AttachBufferAlignmentsTest, AttachesAlignmentToMultipleResults) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t}, {t, t});

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {1, 0});
  std::vector<emitters::KernelArgument> kernel_args;
  kernel_args.reserve(3);
  for (int64_t alignment : {128, 64, 256}) {
    kernel_args.emplace_back(shape);
    kernel_args.back().set_alignment(alignment);
  }

  ASSERT_THAT(AttachBufferAlignments(graph, kernel_args), IsOk());

  auto alignment_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getAlignmentAttrName();
  auto r0 =
      graph.getResultAttrOfType<mlir::IntegerAttr>(0, alignment_attr_name);
  ASSERT_TRUE(r0 != nullptr);
  EXPECT_EQ(r0.getInt(), 64);
  auto r1 =
      graph.getResultAttrOfType<mlir::IntegerAttr>(1, alignment_attr_name);
  ASSERT_TRUE(r1 != nullptr);
  EXPECT_EQ(r1.getInt(), 256);

  graph.erase();
}

TEST_F(AttachBufferAlignmentsTest, ErrorPathKernelArgsCountMismatch) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t0 = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t0}, {t0});

  // Pass 1 argument instead of 2 (1 input + 1 output).
  std::vector<emitters::KernelArgument> kernel_args;
  kernel_args.emplace_back(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {1, 0}));

  EXPECT_THAT(AttachBufferAlignments(graph, kernel_args),
              StatusIs(absl::StatusCode::kInternal));

  graph.erase();
}

// Strides are attached during import and alignments later, in the emitter.
// Both write into the same per-argument attribute dictionary, so verify that
// the second call does not drop what the first one wrote.
TEST_F(AttachBufferAlignmentsTest, ComposesWithLayoutStrides) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY entry {
      p0 = f32[4,16]{0,1} parameter(0)
      ROOT id = f32[4,16]{0,1} copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHloText));

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect>();
  mlir::OpBuilder b(&context);

  auto t = mlir::RankedTensorType::get({4, 16}, b.getF32Type());
  auto graph = MakeGraph(b, {t}, {t});

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 16}, {0, 1});
  std::vector<emitters::KernelArgument> kernel_args;
  kernel_args.reserve(2);
  for (int64_t alignment : {128, 256}) {
    kernel_args.emplace_back(shape);
    kernel_args.back().set_alignment(alignment);
  }

  ASSERT_THAT(AttachLayoutStrides(graph, *hlo_module->entry_computation()),
              IsOk());
  ASSERT_THAT(AttachBufferAlignments(graph, kernel_args), IsOk());

  auto alignment_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getAlignmentAttrName();
  auto stride_attr_name =
      mlir::nv_tensor_ir::TensorIRDialect::getStrideAttrName();

  auto a0_stride =
      graph.getArgAttrOfType<mlir::StringAttr>(0, stride_attr_name);
  ASSERT_TRUE(a0_stride != nullptr);
  EXPECT_EQ(a0_stride.getValue().str(), "(1,4)");
  auto a0_align =
      graph.getArgAttrOfType<mlir::IntegerAttr>(0, alignment_attr_name);
  ASSERT_TRUE(a0_align != nullptr);
  EXPECT_EQ(a0_align.getInt(), 128);

  auto res_stride =
      graph.getResultAttrOfType<mlir::StringAttr>(0, stride_attr_name);
  ASSERT_TRUE(res_stride != nullptr);
  EXPECT_EQ(res_stride.getValue().str(), "(1,4)");
  auto res_align =
      graph.getResultAttrOfType<mlir::IntegerAttr>(0, alignment_attr_name);
  ASSERT_TRUE(res_align != nullptr);
  EXPECT_EQ(res_align.getInt(), 256);

  graph.erase();
}

}  // namespace
}  // namespace xla::gpu::tensor_ir
