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

#include "xla/backends/gpu/codegen/triton/lowering_util.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using ::testing::ElementsAre;

namespace xgt = ::xla::gpu::triton;
namespace xla::gpu {
namespace {

class EmitterHelpersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<
        mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
        mlir::triton::xla::XlaTritonDialect, mlir::LLVM::LLVMDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> ParseModule(
      const std::string& mlir_module) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_module, &context_);
    CHECK(module);
    return module;
  }

  std::pair<mlir::Block*, mlir::OwningOpRef<mlir::ModuleOp>>
  CreateModuleAndEntryBlock(llvm::ArrayRef<mlir::Type> arg_types = {}) {
    mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
        xla::llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&context_));
    mlir::OpBuilder op_builder(mlir_module->getBodyRegion());
    const auto func_type = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(&context_),
        mlir::SmallVector<mlir::Type>(arg_types));
    auto func_op = mlir::LLVM::LLVMFuncOp::create(
        op_builder, mlir::UnknownLoc::get(&context_), "test_func", func_type);
    mlir::Block* entry_block = func_op.addEntryBlock(op_builder);
    return {entry_block, std::move(mlir_module)};
  }

  mlir::MLIRContext context_;
};

TEST_F(EmitterHelpersTest, ExtractTmaMetadataWorksCorrectlyWhenTmaIsUsed) {
  const std::string kMlirModule = R"(
module {
  llvm.func @fusion_impl(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32},
                        %arg1: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [123, 512], tile_shape = [32, 64], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 4, swizzle_mode = "128b">},
                        %arg2: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 512], tile_shape = [16, 64], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 4, swizzle_mode = "128b">},
                        %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>)
                        attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    llvm.return
  }
})";

  mlir::OwningOpRef<mlir::ModuleOp> module = ParseModule(kMlirModule);
  mlir::LLVM::LLVMFuncOp func_op =
      *module->getOps<mlir::LLVM::LLVMFuncOp>().begin();
  TF_ASSERT_OK_AND_ASSIGN(stream_executor::gpu::TmaMetadata tma_metadata,
                          xgt::ExtractTmaMetadata(func_op));

  EXPECT_EQ(tma_metadata.arg_index_to_tma_info.size(), 2);
  EXPECT_TRUE(tma_metadata.arg_index_to_tma_info.contains(1));
  EXPECT_TRUE(tma_metadata.arg_index_to_tma_info.contains(2));

  auto tma_arg_1 = tma_metadata.arg_index_to_tma_info.at(1);
  auto tma_arg_2 = tma_metadata.arg_index_to_tma_info.at(2);

  EXPECT_THAT(tma_arg_1.global_dims(), ElementsAre(512, 123));
  EXPECT_THAT(tma_arg_1.box_dims(), ElementsAre(32, 32));
  EXPECT_THAT(tma_arg_1.element_strides(), ElementsAre(1, 1));
  EXPECT_THAT(tma_arg_1.element_size(), 4);
  EXPECT_EQ(tma_arg_1.swizzle(),
            stream_executor::gpu::TmaDescriptor::TmaSwizzle::k128B);

  EXPECT_THAT(tma_arg_2.global_dims(), ElementsAre(512, 32));
  EXPECT_THAT(tma_arg_2.box_dims(), ElementsAre(32, 16));
  EXPECT_THAT(tma_arg_2.element_strides(), ElementsAre(1, 1));
  EXPECT_THAT(tma_arg_2.element_size(), 4);
  EXPECT_EQ(tma_arg_2.swizzle(),
            stream_executor::gpu::TmaDescriptor::TmaSwizzle::k128B);
}

TEST_F(EmitterHelpersTest, ExtractThreadDimsWorksCorrectlyWithValidInput) {
  const std::string kMlirModule = R"(
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 10240 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  llvm.func @fusion_impl() attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
   llvm.return
  }
})";

  mlir::OwningOpRef<mlir::ModuleOp> module = ParseModule(kMlirModule);
  mlir::LLVM::LLVMFuncOp func_op =
      *module->getOps<mlir::LLVM::LLVMFuncOp>().begin();
  TF_ASSERT_OK_AND_ASSIGN(stream_executor::ThreadDim thread_dims,
                          xgt::ExtractThreadDims(module.get(), func_op));
  EXPECT_EQ(thread_dims, stream_executor::ThreadDim(32, 1, 1));
}

TEST_F(EmitterHelpersTest, ExpandAndBroadcastValueWorksCorrectly) {
  auto [entry_block, mlir_module] = CreateModuleAndEntryBlock();
  mlir::ImplicitLocOpBuilder builder(mlir_module->getLoc(), entry_block,
                                     entry_block->begin());
  const mlir::Type i32_type = builder.getI32Type();
  const auto tile_2d_type = mlir::RankedTensorType::get({16, 32}, i32_type);

  const auto row_type = mlir::RankedTensorType::get({16}, i32_type);
  const auto row_value =
      mlir::triton::MakeRangeOp::create(builder, row_type, 0, 16).getResult();
  // 2D broadcast (dim 0)
  const auto broadcast_0 =
      xgt::ExpandAndBroadcastValue(builder, row_value, 0, tile_2d_type);
  EXPECT_EQ(broadcast_0.getType(), tile_2d_type);

  // 2D broadcast (dim 1)
  const auto col_type = mlir::RankedTensorType::get({32}, i32_type);
  const auto col_value =
      mlir::triton::MakeRangeOp::create(builder, col_type, 0, 32).getResult();
  const auto broadcast_1 =
      xgt::ExpandAndBroadcastValue(builder, col_value, 1, tile_2d_type);
  EXPECT_EQ(broadcast_1.getType(), tile_2d_type);

  // 3D broadcast (dim 1)
  const auto tile_3d_type = mlir::RankedTensorType::get({8, 16, 32}, i32_type);
  const auto mid_dim_value =
      mlir::triton::MakeRangeOp::create(builder, row_type, 0, 16).getResult();
  const auto broadcast_3d =
      xgt::ExpandAndBroadcastValue(builder, mid_dim_value, 1, tile_3d_type);
  EXPECT_EQ(broadcast_3d.getType(), tile_3d_type);

  // 4. 1D broadcast (identity-like)
  const auto broadcast_1d =
      xgt::ExpandAndBroadcastValue(builder, row_value, 0, row_type);
  EXPECT_EQ(broadcast_1d.getType(), row_type);

  // Check that we have the expected number of ops.
  int expand_dims_count = 0;
  int broadcast_count = 0;
  for (mlir::Operation& op : entry_block->getOperations()) {
    if (mlir::isa<mlir::triton::ExpandDimsOp>(op)) {
      expand_dims_count++;
    }
    if (mlir::isa<mlir::triton::BroadcastOp>(op)) {
      broadcast_count++;
    }
  }
  // dim 0 -> 1 expand_dims, 1 broadcast
  // dim 1 -> 1 expand_dims, 1 broadcast
  // 3D dim 1 -> 2 expand_dims, 1 broadcast
  // 1D -> 0 expand_dims, 1 broadcast
  EXPECT_EQ(expand_dims_count, 1 + 1 + 2 + 0);
  EXPECT_EQ(broadcast_count, 4);
}

TEST_F(EmitterHelpersTest, CreateTensorOfPointersAndMask1DNoMask) {
  auto [entry_block, mlir_module] = CreateModuleAndEntryBlock(
      {mlir::LLVM::LLVMPointerType::get(&context_, 1)});
  mlir::ImplicitLocOpBuilder builder(mlir_module->getLoc(), entry_block,
                                     entry_block->begin());

  mlir::Value base_ptr = entry_block->getArgument(0);
  mlir::Value offset =
      builder.create<mlir::arith::ConstantIndexOp>(0).getResult();

  auto [tile, mask_tile] = xgt::CreateTensorOfPointersAndMask(
      builder, base_ptr, /*original_shape=*/{128}, /*layout=*/{0},
      /*offsets=*/{offset}, /*sizes=*/{128}, /*strides=*/{1},
      /*reduced_dims=*/{}, /*tile_shape=*/{128});

  ASSERT_TRUE(tile);
  EXPECT_FALSE(mask_tile);
  const auto tile_type = mlir::cast<mlir::RankedTensorType>(tile.getType());
  EXPECT_THAT(tile_type.getShape(), ElementsAre(128));
  EXPECT_EQ(tile_type.getElementType(), base_ptr.getType());
}

TEST_F(EmitterHelpersTest, CreateTensorOfPointersAndMask1DWithMask) {
  auto [entry_block, mlir_module] = CreateModuleAndEntryBlock(
      {mlir::LLVM::LLVMPointerType::get(&context_, 1)});
  mlir::ImplicitLocOpBuilder builder(mlir_module->getLoc(), entry_block,
                                     entry_block->begin());

  mlir::Value base_ptr = entry_block->getArgument(0);
  mlir::Value offset =
      builder.create<mlir::arith::ConstantIndexOp>(0).getResult();

  auto [tile, mask] = xgt::CreateTensorOfPointersAndMask(
      builder, base_ptr, /*original_shape=*/{100}, /*layout=*/{0},
      /*offsets=*/{offset}, /*sizes=*/{128}, /*strides=*/{1},
      /*reduced_dims=*/{}, /*tile_shape=*/{128});

  ASSERT_TRUE(tile);
  ASSERT_TRUE(mask);
  const auto mask_tile_type =
      mlir::cast<mlir::RankedTensorType>(mask.getType());
  EXPECT_THAT(mask_tile_type.getShape(), ElementsAre(128));
  EXPECT_TRUE(mask_tile_type.getElementType().isInteger(1));
}

TEST_F(EmitterHelpersTest, CreateTensorOfPointersAndMask2DWithReducedDim) {
  auto [entry_block, mlir_module] = CreateModuleAndEntryBlock(
      {mlir::LLVM::LLVMPointerType::get(&context_, 1)});
  mlir::ImplicitLocOpBuilder builder(mlir_module->getLoc(), entry_block,
                                     entry_block->begin());

  mlir::Value base_ptr = entry_block->getArgument(0);
  mlir::Value offset_0 =
      builder.create<mlir::arith::ConstantIndexOp>(0).getResult();
  mlir::Value offset_1 =
      builder.create<mlir::arith::ConstantIndexOp>(0).getResult();

  auto [tile, mask_tile] = xgt::CreateTensorOfPointersAndMask(
      builder, base_ptr, /*original_shape=*/{64, 128}, /*layout=*/{1, 0},
      /*offsets=*/{offset_0, offset_1}, /*sizes=*/{1, 128}, /*strides=*/{1, 1},
      /*reduced_dims=*/{0}, /*tile_shape=*/{128});

  ASSERT_TRUE(tile);
  EXPECT_FALSE(mask_tile);
  const auto tile_type = mlir::cast<mlir::RankedTensorType>(tile.getType());
  EXPECT_THAT(tile_type.getShape(), ElementsAre(128));
}

TEST_F(EmitterHelpersTest, CreateTensorOfPointersAndMaskWithTensorBasePtr) {
  auto [entry_block, mlir_module] = CreateModuleAndEntryBlock(
      {mlir::LLVM::LLVMPointerType::get(&context_, 1)});
  mlir::ImplicitLocOpBuilder builder(mlir_module->getLoc(), entry_block,
                                     entry_block->begin());

  mlir::Value scalar_base_ptr = entry_block->getArgument(0);
  auto tile_type =
      mlir::RankedTensorType::get({128}, scalar_base_ptr.getType());
  const mlir::Value base_ptr =
      mlir::triton::SplatOp::create(builder, tile_type, scalar_base_ptr);
  const mlir::Value offset =
      mlir::arith::ConstantIndexOp::create(builder, 1024).getResult();
  auto [ptr_tile, mask_tile] = xgt::CreateTensorOfPointersAndMask(
      builder, base_ptr, /*original_shape=*/{2048}, /*layout=*/{0},
      /*offsets=*/{offset}, /*sizes=*/{128}, /*strides=*/{1},
      /*reduced_dims=*/{}, /*tile_shape=*/{128});
  ASSERT_TRUE(ptr_tile);
  EXPECT_FALSE(mask_tile);
  EXPECT_EQ(ptr_tile.getType(), tile_type);
}
}  // namespace
}  // namespace xla::gpu
