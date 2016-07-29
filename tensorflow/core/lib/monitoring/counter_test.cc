/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/monitoring/counter.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

class LabeledCounterTest : public ::testing::Test {
 protected:
  LabeledCounterTest() {}

  Counter<1> counter_with_labels_{{"/tensorflow/test/counter_with_labels_",
                                   "Counter with one label.", "One label"}};
};

TEST_F(LabeledCounterTest, InitializedWithZero) {
  EXPECT_EQ(0, counter_with_labels_.GetCell("Empty")->value());
}

TEST_F(LabeledCounterTest, GetCell) {
  auto* cell = counter_with_labels_.GetCell("GetCellOp");
  EXPECT_EQ(0, cell->value());

  cell->IncrementBy(42);
  EXPECT_EQ(42, cell->value());

  auto* same_cell = counter_with_labels_.GetCell("GetCellOp");
  EXPECT_EQ(42, same_cell->value());

  same_cell->IncrementBy(58);
  EXPECT_EQ(100, cell->value());
  EXPECT_EQ(100, same_cell->value());
}

using LabeledCounterDeathTest = LabeledCounterTest;

TEST_F(LabeledCounterDeathTest, DiesOnDecrement) {
  EXPECT_DEBUG_DEATH(
      { counter_with_labels_.GetCell("DyingOp")->IncrementBy(-1); },
      "decrement");
}

class UnlabeledCounterTest : public ::testing::Test {
 protected:
  UnlabeledCounterTest() {}

  Counter<0> counter_without_labels_{
      {"/tensorflow/test/counter0", "Counter without any labels."}};
};

TEST_F(UnlabeledCounterTest, InitializedWithZero) {
  EXPECT_EQ(0, counter_without_labels_.GetCell()->value());
}

TEST_F(UnlabeledCounterTest, GetCell) {
  auto* cell = counter_without_labels_.GetCell();
  EXPECT_EQ(0, cell->value());

  cell->IncrementBy(42);
  EXPECT_EQ(42, cell->value());

  auto* same_cell = counter_without_labels_.GetCell();
  EXPECT_EQ(42, same_cell->value());

  same_cell->IncrementBy(58);
  EXPECT_EQ(100, cell->value());
  EXPECT_EQ(100, same_cell->value());
}

using UnlabeledCounterDeathTest = UnlabeledCounterTest;

TEST_F(UnlabeledCounterDeathTest, DiesOnDecrement) {
  EXPECT_DEBUG_DEATH({ counter_without_labels_.GetCell()->IncrementBy(-1); },
                     "decrement");
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
