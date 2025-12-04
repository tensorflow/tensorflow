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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
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
}  // namespace
}  // namespace xla::gpu
