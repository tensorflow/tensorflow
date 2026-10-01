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

#include "xla/service/gpu/mlir_context_pool.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;

TEST(MlirContextPoolTest, ReturnsFallbackWhenPoolIsNull) {
  mlir::MLIRContext fallback;
  ASSERT_OK_AND_ASSIGN(PooledOrFallbackMlirContext context,
                       BorrowMlirContextOr(/*pool=*/nullptr, &fallback));
  EXPECT_EQ(context.get(), &fallback);
}

TEST(MlirContextPoolTest, BorrowsFromPoolAndReturnsOnDestruction) {
  MlirContextPool pool([] { return std::make_unique<mlir::MLIRContext>(); },
                       /*preallocate=*/1);
  mlir::MLIRContext fallback;
  {
    ASSERT_OK_AND_ASSIGN(PooledOrFallbackMlirContext context,
                         BorrowMlirContextOr(&pool, &fallback));
    EXPECT_NE(context.get(), nullptr);
    EXPECT_NE(context.get(), &fallback);
    EXPECT_EQ(pool.num_available(), 0);
  }
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_created(), 1);
}

TEST(MlirContextPoolTest, MovedFromContextKeepsBorrowAlive) {
  MlirContextPool pool([] { return std::make_unique<mlir::MLIRContext>(); },
                       /*preallocate=*/1);
  {
    ASSERT_OK_AND_ASSIGN(PooledOrFallbackMlirContext context,
                         BorrowMlirContextOr(&pool, /*fallback=*/nullptr));
    mlir::MLIRContext* raw = context.get();
    PooledOrFallbackMlirContext moved = std::move(context);
    EXPECT_EQ(moved.get(), raw);
    EXPECT_EQ(pool.num_available(), 0);
  }
  EXPECT_EQ(pool.num_available(), 1);
}

TEST(MlirContextPoolTest, PropagatesBuilderError) {
  MlirContextPool pool(
      []() -> absl::StatusOr<std::unique_ptr<mlir::MLIRContext>> {
        return absl::InternalError("builder failed");
      });
  mlir::MLIRContext fallback;
  EXPECT_THAT(BorrowMlirContextOr(&pool, &fallback),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xla::gpu
