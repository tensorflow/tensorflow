//===- DialectTest.cpp - Dialect unit tests -------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
struct TestDialect : public Dialect {
  TestDialect(MLIRContext *context) : Dialect(/*namePrefix=*/"test", context) {}
};

TEST(DialectDeathTest, MultipleDialectsWithSameNamespace) {
  MLIRContext context;

  // Registering a dialect with the same namespace twice should result in a
  // failure.
  new TestDialect(&context);
  ASSERT_DEATH(new TestDialect(&context), "");
}

} // end namespace
