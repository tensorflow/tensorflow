//===- RulesTest.cpp - Rules unit tests -----------------------------------===//
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

#include "mlir/Quantizer/Support/Rules.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::quantizer;

namespace {

using TestDiscreteFact = DiscreteFact<int>;

TEST(ExpandingMinMaxReducer, Basic) {
  ExpandingMinMaxFact f;
  EXPECT_FALSE(f.hasValue());

  // First assertion always modifies.
  EXPECT_TRUE(modified(f.assertValue(0, {-1.0, 1.0})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-1.0, 1.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  // Assertion in the same band expands.
  EXPECT_TRUE(modified(f.assertValue(0, {-0.5, 2.0})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-1.0, 2.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  EXPECT_TRUE(modified(f.assertValue(0, {-2.0, 0.5})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-2.0, 2.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  // Same band smaller bound does not modify.
  EXPECT_FALSE(modified(f.assertValue(0, {-0.5, 0.5})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-2.0, 2.0), f.getValue());
  EXPECT_EQ(0, f.getSalience());

  // Higher salience overrides.
  EXPECT_TRUE(modified(f.assertValue(10, {-0.2, 0.2})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f.getValue());
  EXPECT_EQ(10, f.getSalience());

  // Lower salience no-ops.
  EXPECT_FALSE(modified(f.assertValue(5, {-2.0, 2.0})));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f.getValue());
  EXPECT_EQ(10, f.getSalience());

  // Merge from a fact without a value no-ops.
  ExpandingMinMaxFact f1;
  EXPECT_FALSE(modified(f.mergeFrom(f1)));
  EXPECT_TRUE(f.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f.getValue());
  EXPECT_EQ(10, f.getSalience());

  // Merge from a fact with a value merges.
  EXPECT_TRUE(modified(f1.mergeFrom(f)));
  EXPECT_TRUE(f1.hasValue());
  EXPECT_EQ(std::make_pair(-0.2, 0.2), f1.getValue());
  EXPECT_EQ(10, f1.getSalience());
}

TEST(TestDiscreteFact, Basic) {
  TestDiscreteFact f;
  EXPECT_FALSE(f.hasValue());

  // Initial value.
  EXPECT_TRUE(modified(f.assertValue(0, {2})));
  EXPECT_FALSE(modified(f.assertValue(0, {2})));
  EXPECT_EQ(2, f.getValue().value);
  EXPECT_FALSE(f.getValue().conflict);

  // Conflicting update.
  EXPECT_TRUE(modified(f.assertValue(0, {4})));
  EXPECT_EQ(2, f.getValue().value); // Arbitrary but known to be first wins.
  EXPECT_TRUE(f.getValue().conflict);

  // Further update still conflicts.
  EXPECT_FALSE(modified(f.assertValue(0, {4})));
  EXPECT_EQ(2, f.getValue().value); // Arbitrary but known to be first wins.
  EXPECT_TRUE(f.getValue().conflict);

  // Different salience update does not conflict.
  EXPECT_TRUE(modified(f.assertValue(1, {6})));
  EXPECT_EQ(6, f.getValue().value);
  EXPECT_FALSE(f.getValue().conflict);
}

} // end anonymous namespace
