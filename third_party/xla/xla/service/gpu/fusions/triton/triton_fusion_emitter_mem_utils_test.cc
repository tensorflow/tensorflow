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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/fusions/triton/triton_fusion_emitter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu::ir_emitter_triton_internal {
namespace {

using ::llvm::SmallVector;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::Type;
using ::mlir::Value;
using ::testing::ElementsAre;

class TritonMakeTensorPtrTest : public HloTestBase {
 public:
  void SetUp() override { LoadMlirDialectsForTriton(mlir_context_); }

  std::pair<std::unique_ptr<VerifiedHloModule>, TiledHloComputation>
  CreateAndTileParameterHloInstruction(
      std::vector<int64_t> shape_sizes, const std::vector<int64_t>& tile_sizes,
      const std::vector<int64_t>& tile_strides);

  std::pair<mlir::OwningOpRef<mlir::ModuleOp>, MakeTensorPtrOpAndBoundaryChecks>
  CreateTestTensorPtr(const std::vector<int64_t>& parent_shape,
                      const std::vector<int64_t>& tile_sizes,
                      const std::vector<int64_t>& tile_strides);

 protected:
  MLIRContext mlir_context_;
};

// Returns a Parameter HLO instruction with a parameter number 0.
std::pair<std::unique_ptr<VerifiedHloModule>, TiledHloComputation>
TritonMakeTensorPtrTest::CreateAndTileParameterHloInstruction(
    std::vector<int64_t> shape_sizes, const std::vector<int64_t>& tile_sizes,
    const std::vector<int64_t>& tile_strides) {
  const std::string hlo_text = R"(
  HloModule test_module

  fusion {
    p0 = f32[$0] parameter(0)
    ROOT log = f32[$0] log(p0)
  }

  ENTRY %main{
    p0.1 = f32[$0] parameter(0)
    ROOT fusion = f32[$0] fusion(p0.1), kind=kLoop, calls=%fusion
  })";
  auto verified_hlo_module_or = ParseAndReturnVerifiedModule(
      absl::Substitute(hlo_text, absl::StrJoin(shape_sizes, ",")));
  CHECK_OK(verified_hlo_module_or);

  std::unique_ptr<VerifiedHloModule> verified_hlo_module =
      std::move(verified_hlo_module_or).value();

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(
      verified_hlo_module->entry_computation()->root_instruction());

  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeFusion(
          *fusion_adaptor, &mlir_context_,
          TritonEmitterConstraints::GetBuilder(
              TestGpuDeviceInfo::RTXA6000DeviceInfo()));
  CHECK(
      std::holds_alternative<SymbolicTileAnalysis>(symbolic_tile_analysis_or));

  SymbolicTileAnalysis symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(std::move(symbolic_tile_analysis_or));

  auto tiled_hlo_computation_or =
      symbolic_tile_analysis.ComputeTiledHloInstructions(
          tile_sizes, /*constraints_are_known_satisfied=*/true,
          /*compute_all_tile_offset_indexing_maps=*/true);
  TF_EXPECT_OK(tiled_hlo_computation_or.status());
  return std::make_pair(std::move(verified_hlo_module),
                        *std::move(tiled_hlo_computation_or));
}

mlir::triton::FuncOp CreateTritonFunction(
    ImplicitLocOpBuilder& b, const std::vector<int64_t> shape_sizes) {
  auto fn = b.create<::mlir::triton::FuncOp>(
      "func",
      b.getFunctionType({::mlir::triton::PointerType::get(
                            b.getF32Type(), mlir::NVVM::kGlobalMemorySpace)},
                        std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  b.setInsertionPointToStart(fn.addEntryBlock());
  return fn;
}

std::pair<mlir::OwningOpRef<mlir::ModuleOp>, MakeTensorPtrOpAndBoundaryChecks>
TritonMakeTensorPtrTest::CreateTestTensorPtr(
    const std::vector<int64_t>& parent_shape,
    const std::vector<int64_t>& tile_sizes,
    const std::vector<int64_t>& tile_strides) {
  auto [hlo_module, tiled_hlo_computation] =
      CreateAndTileParameterHloInstruction(parent_shape, tile_sizes,
                                           tile_strides);

  const TiledHloInstruction* tiled_hlo =
      tiled_hlo_computation.GetRoot()->operand(0);
  const HloInstruction* hlo = tiled_hlo->hlo();

  OpBuilder builder(&mlir_context_);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(hlo->name()));
  mlir::OwningOpRef<mlir::ModuleOp> triton_module =
      llvm_ir::CreateMlirModuleOp(loc);
  builder.setInsertionPointToEnd(triton_module->getBody());

  ImplicitLocOpBuilder b(loc, builder);
  auto fn = CreateTritonFunction(b, parent_shape);

  SmallVector<Value, 3> tile_multi_index = ComputeDelinearizedTileIndex(
      b, tiled_hlo_computation.num_output_tiles_per_dim());

  return std::make_pair(
      std::move(triton_module),
      *ir_emitter_triton_internal::CreateMakeTensorPtrOp(
          b, tile_multi_index, *tiled_hlo, fn.getArgument(0)));
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

mlir::ArrayRef<int64_t> TensorShape(const ::mlir::triton::MakeTensorPtrOp& op) {
  auto ptr =
      mlir::cast<::mlir::triton::PointerType>(op->getResult(0).getType());
  auto tensor = mlir::cast<mlir::TensorType>(ptr.getPointeeType());
  return tensor.getShape();
}

void CheckSizesAreSubtractions(const mlir::ValueRange size_values) {
  for (Value v : size_values) {
    EXPECT_NE(v.getDefiningOp<mlir::arith::SubIOp>(), nullptr);
  }
}
TEST_F(TritonMakeTensorPtrTest, BlockProperties) {
  {
    auto [module, ptr] = CreateTestTensorPtr({15, 20}, {3, 4}, {1, 1});
    CheckSizesAreSubtractions(ptr.op.getShape());
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(4, 4));
    EXPECT_THAT(ptr.boundary_checks, ElementsAre(0));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(20, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(1, 0));
  }
  {
    auto [module, ptr] = CreateTestTensorPtr({20, 20}, {4, 4}, {1, 1});
    CheckSizesAreSubtractions(ptr.op.getShape());
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(4, 4));
    EXPECT_TRUE(ptr.boundary_checks.empty());
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(20, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(1, 0));
  }
  {
    auto [module, ptr] = CreateTestTensorPtr({5}, {1}, {1});
    CheckSizesAreSubtractions(ptr.op.getShape());
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(1));
    EXPECT_TRUE(ptr.boundary_checks.empty());
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(0));
  }
  {
    auto [module, ptr] = CreateTestTensorPtr({5, 5, 5}, {1, 1, 1}, {1, 1, 1});
    CheckSizesAreSubtractions(ptr.op.getShape());
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(1, 1, 1));
    EXPECT_TRUE(ptr.boundary_checks.empty());
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()), ElementsAre(25, 5, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(2, 1, 0));
  }
  {
    auto [module, ptr] = CreateTestTensorPtr({5, 15, 20}, {1, 3, 4}, {1, 1, 1});
    CheckSizesAreSubtractions(ptr.op.getShape());
    EXPECT_THAT(TensorShape(ptr.op), ElementsAre(1, 4, 4));
    EXPECT_THAT(ptr.boundary_checks, ElementsAre(1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getStrides()),
                ElementsAre(300, 20, 1));
    EXPECT_THAT(ConstOpValuesToInt(ptr.op.getOffsets()), ElementsAre(0, 0, 0));
    EXPECT_THAT(ptr.op.getOrder(), ElementsAre(2, 1, 0));
  }
}

}  // namespace
}  // namespace xla::gpu::ir_emitter_triton_internal
