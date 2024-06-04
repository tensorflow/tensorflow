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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "third_party/triton/include/triton/Dialect/Triton/IR/Dialect.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu::ir_emitter_triton_internal {
namespace {

using ::mlir::ImplicitLocOpBuilder;
using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::Type;
using ::mlir::Value;
using ::testing::ElementsAre;

class TritonMakeTensorPtrTest : public HloTestBase {
 public:
  void SetUp() override { LoadMlirDialectsForTriton(mlir_context_); }

 protected:
  MLIRContext mlir_context_;
};

// This is not a proper affine map, just something that enables the
// creation of the index.
//
// TODO(b/332649307): Test with a proper affine map once the code starts
// passing proper offsets to MakeTensorPtr.
IndexingMap CreateAffineMap(const std::vector<int64_t>& tile_sizes,
                            MLIRContext& ctx) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &ctx);
  std::vector<mlir::AffineExpr> dims(tile_sizes.size(), d0);
  return IndexingMap::FromTensorSizes(mlir::AffineMap::get(1, 0, dims, &ctx),
                                      /*dim_upper_bounds=*/{8},
                                      /*symbol_upper_bounds=*/{});
}

// Returns a Parameter HLO instruction with a parameter number 0.
std::pair<std::unique_ptr<HloInstruction>, std::unique_ptr<TiledHloInstruction>>
CreateAndTileParameterHloInstruction(std::vector<int64_t> shape_sizes,
                                     const std::vector<int64_t>& tile_sizes,
                                     const std::vector<int64_t>& tile_strides,
                                     MLIRContext& ctx) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, shape_sizes), "p0");

  auto tiled_hlo = TiledHloInstruction::Create(
      hlo.get(), tile_sizes, tile_strides, CreateAffineMap(tile_sizes, ctx));
  EXPECT_OK(tiled_hlo);
  return std::make_pair(std::move(hlo), std::move(tiled_hlo.value()));
}

mlir::triton::FuncOp CreateTritonFunction(
    ImplicitLocOpBuilder& b, const std::vector<int64_t> shape_sizes) {
  auto fn = b.create<mt::FuncOp>(
      "func",
      b.getFunctionType({mt::PointerType::get(b.getF32Type(),
                                              mlir::NVVM::kGlobalMemorySpace)},
                        std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  b.setInsertionPointToStart(fn.addEntryBlock());
  return fn;
}

std::pair<mlir::OwningOpRef<mlir::ModuleOp>, MakeTensorPtrOpAndBoundaryChecks>
CreateTestTensorPtr(const std::vector<int64_t>& tile_sizes,
                    const std::vector<int64_t>& tile_strides,
                    MLIRContext& ctx) {
  std::vector<int64_t> shape_sizes;
  for (int64_t tile_size : tile_sizes) {
    constexpr int64_t kShapeToTileRatio = 5;
    shape_sizes.push_back(tile_size * kShapeToTileRatio);
  }

  auto [hlo, tiled_hlo] = CreateAndTileParameterHloInstruction(
      shape_sizes, tile_sizes, tile_strides, ctx);

  OpBuilder builder(&ctx);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(hlo->name()));
  mlir::OwningOpRef<mlir::ModuleOp> triton_module =
      llvm_ir::CreateMlirModuleOp(loc);
  builder.setInsertionPointToEnd(triton_module->getBody());

  ImplicitLocOpBuilder b(loc, builder);
  auto fn = CreateTritonFunction(b, shape_sizes);
  Value pid = b.create<mlir::arith::IndexCastUIOp>(
      b.getIndexType(), b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::X));
  return std::make_pair(std::move(triton_module),
                        ir_emitter_triton_internal::CreateMakeTensorPtrOp(
                            b, pid, *tiled_hlo, fn.getArgument(0)));
}

std::vector<int> ConstOpValuesToInt(const mlir::ValueRange values) {
  std::vector<int> result;
  for (Value v : values) {
    auto const_op = v.getDefiningOp<mlir::arith::ConstantOp>();
    CHECK_NOTNULL(const_op);
    auto int_attr = mlir::cast<mlir::IntegerAttr>(const_op.getValueAttr());
    result.push_back(int_attr.getInt());
  }
  return result;
}

mlir::ArrayRef<int64_t> TensorShape(const mt::MakeTensorPtrOp& op) {
  auto ptr = mlir::cast<mt::PointerType>(op->getResult(0).getType());
  auto tensor = mlir::cast<mlir::TensorType>(ptr.getPointeeType());
  return tensor.getShape();
}

TEST_F(TritonMakeTensorPtrTest, BlockProperties) {
  {
    auto [module, ptr] = CreateTestTensorPtr({3, 4}, {1, 1}, mlir_context_);
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getShape()), ElementsAre(3, 4));
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(4, 4));
    EXPECT_THAT(ptr.boundary_checks, ElementsAre(0));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(1, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(1, 0));
  }
  {
    auto [module, ptr] = CreateTestTensorPtr({4, 4}, {1, 1}, mlir_context_);
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getShape()), ElementsAre(4, 4));
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(4, 4));
    EXPECT_TRUE(ptr.boundary_checks.empty());
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(1, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(1, 0));
  }
  {
    auto [module, ptr] = CreateTestTensorPtr({1}, {1}, mlir_context_);
    EXPECT_TRUE(ConstOpValuesToInt(ptr.op.getShape()).empty());
    EXPECT_TRUE(TensorShape(ptr.op).empty());
    EXPECT_TRUE(ptr.boundary_checks.empty());
    EXPECT_TRUE(ConstOpValuesToInt(ptr.op.getStrides()).empty());
    EXPECT_TRUE(ConstOpValuesToInt(ptr.op.getOffsets()).empty());
    EXPECT_TRUE(ptr.op.getOrder().empty());
  }
  {
    auto [module, ptr] =
        CreateTestTensorPtr({1, 1, 1}, {1, 1, 1}, mlir_context_);
    EXPECT_TRUE(ConstOpValuesToInt(ptr.op.getShape()).empty());
    EXPECT_TRUE(TensorShape(ptr.op).empty());
    EXPECT_TRUE(ptr.boundary_checks.empty());
    EXPECT_TRUE(ConstOpValuesToInt(ptr.op.getStrides()).empty());
    EXPECT_TRUE(ConstOpValuesToInt(ptr.op.getOffsets()).empty());
    EXPECT_TRUE(ptr.op.getOrder().empty());
  }
  {
    auto [module, ptr] =
        CreateTestTensorPtr({1, 3, 4}, {1, 1, 1}, mlir_context_);
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getShape()), ElementsAre(3, 4));
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(4, 4));
    EXPECT_THAT(ptr.boundary_checks, ElementsAre(0));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(1, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(1, 0));
  }
  {
    // TODO(b/332649307): Clarify whether the 1 at index 3 should indeed be
    // skipped. Maybe this depends on the shape? E.g. if the shape is also 1,
    // then it's fine to skip, otherwise not.
    auto [module, ptr] =
        CreateTestTensorPtr({1, 3, 4, 1, 6}, {1, 1, 1, 1, 1}, mlir_context_);
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getShape()), ElementsAre(3, 4, 6));
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(4, 4, 8));
    EXPECT_THAT(ptr.boundary_checks, ElementsAre(0, 2));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(1, 1, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(2, 1, 0));
  }
}

}  // namespace
}  // namespace xla::gpu::ir_emitter_triton_internal
