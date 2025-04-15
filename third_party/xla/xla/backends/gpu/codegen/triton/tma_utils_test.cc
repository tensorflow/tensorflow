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

#include "xla/backends/gpu/codegen/triton/tma_utils.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::absl::StatusCode;
using ::llvm::SmallVector;
using ::stream_executor::gpu::TmaDescriptor;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(Create2DTmaDescriptorTest, ValidInputReturnCorrectDescriptor) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  llvm::SmallVector<int64_t, 2> global_shape = {256, 128};
  llvm::SmallVector<int64_t, 2> block_shape = {64, 32};
  int element_byte_size = 4;
  TF_ASSERT_OK_AND_ASSIGN(
      TmaDescriptor tma_desc,
      Create2DTmaDescriptor(global_shape, block_shape, element_byte_size));
  EXPECT_EQ(tma_desc.element_size(), 4);
  EXPECT_EQ(tma_desc.num_dimensions(), 2);
  EXPECT_THAT(tma_desc.global_dims(), ElementsAre(128, 256));
  EXPECT_THAT(tma_desc.global_strides(), ElementsAre(128 * 4));
  EXPECT_THAT(tma_desc.box_dims(), ElementsAre(32, 64));
  EXPECT_THAT(tma_desc.element_strides(), ElementsAre(1, 1));
  EXPECT_EQ(tma_desc.interleave(), TmaDescriptor::TmaInterleave::kNone);
  EXPECT_EQ(tma_desc.swizzle(), TmaDescriptor::TmaSwizzle::k128B);
  EXPECT_EQ(tma_desc.l2_promotion(), TmaDescriptor::TmaL2Promotion::k128B);
  EXPECT_EQ(tma_desc.float_oob_fill(), TmaDescriptor::TmaFloatOobFill::kNone);
}

TEST(Create2DTmaDescriptorTest, BadGlobalShapeFailsGracefully) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  llvm::SmallVector<int64_t, 1> global_shape = {128};
  llvm::SmallVector<int64_t, 2> block_shape = {128, 128};
  int element_byte_size = 4;
  EXPECT_THAT(
      Create2DTmaDescriptor(global_shape, block_shape, element_byte_size),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("expected 2D global shape")));
}

TEST(Create2DTmaDescriptorTest, BadBlockShapeFailsGracefully) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  llvm::SmallVector<int64_t, 2> global_shape = {128, 128};
  llvm::SmallVector<int64_t, 2> block_shape = {128};
  int element_byte_size = 4;
  EXPECT_THAT(
      Create2DTmaDescriptor(global_shape, block_shape, element_byte_size),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("expected 2D block shape")));
}

TEST(Create2DTmaDescriptorTest, SmallBlockShapeFailsGracefully) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  llvm::SmallVector<int64_t, 2> global_shape = {128, 128};
  llvm::SmallVector<int64_t, 2> block_shape = {128, 2};
  int element_byte_size = 4;
  EXPECT_THAT(
      Create2DTmaDescriptor(global_shape, block_shape, element_byte_size),
      StatusIs(StatusCode::kFailedPrecondition,
               HasSubstr("dimension size too small")));
}

}  // namespace
}  // namespace xla::gpu
