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

#include "xla/service/gpu/runtime/tma_metadata.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {
namespace {

using absl::StatusCode;

TEST(TmaMetadataTest, CreateValidTmaInfo) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  ASSERT_TRUE(TmaInfo::Create({128, 128}, {500, 360}, b.getBF16Type()).ok());
  ASSERT_TRUE(TmaInfo::Create({256, 128}, {480, 360}, b.getF32Type()).ok());
}

TEST(TmaMetadataTest, CreateInvalidDims) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  EXPECT_EQ(TmaInfo::Create({128, 128, 128}, {500, 360, 280}, b.getBF16Type())
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(TmaInfo::Create({128, 128, 128}, {500, 360}, b.getBF16Type())
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(TmaInfo::Create({128}, {360}, b.getBF16Type()).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(TmaMetadataTest, CreateInvalidTensorSize) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  EXPECT_EQ(
      TmaInfo::Create({128, 512}, {360, 280}, b.getBF16Type()).status().code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      TmaInfo::Create({512, 64}, {360, 280}, b.getBF16Type()).status().code(),
      StatusCode::kInvalidArgument);
}

TEST(TmaMetadataTest, CreateInvalidElementType) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  EXPECT_EQ(
      TmaInfo::Create({256, 128}, {480, 360}, b.getI4Type()).status().code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      TmaInfo::Create({256, 128}, {480, 360}, b.getF64Type()).status().code(),
      StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace xla::gpu
