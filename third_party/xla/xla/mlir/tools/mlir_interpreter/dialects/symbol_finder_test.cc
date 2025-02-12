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

#include "xla/mlir/tools/mlir_interpreter/dialects/symbol_finder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace mlir::interpreter {

// This defines the symbol we're looking for in the test.

namespace {
using ::testing::NotNull;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(SymbolFinderTest, FindSymbolInProcess) {
  // `malloc` should be available on every platform
  EXPECT_THAT(FindSymbolInProcess("malloc"), IsOkAndHolds(NotNull()));
  EXPECT_THAT(FindSymbolInProcess("surely_this_does_not_exist"),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace mlir::interpreter
