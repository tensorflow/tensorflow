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

auto* counter_with_labels =
    Counter<1>::New("/tensorflow/test/counter_with_labels",
                    "Counter with one label.", "MyLabel");

TEST(LabeledCounterTest, InitializedWithZero) {
  EXPECT_EQ(0, counter_with_labels->GetCell("Empty")->value());
}

TEST(LabeledCounterTest, GetCell) {
  auto* cell = counter_with_labels->GetCell("GetCellOp");
  EXPECT_EQ(0, cell->value());

  cell->IncrementBy(42);
  EXPECT_EQ(42, cell->value());

  auto* same_cell = counter_with_labels->GetCell("GetCellOp");
  EXPECT_EQ(42, same_cell->value());

  same_cell->IncrementBy(58);
  EXPECT_EQ(100, cell->value());
  EXPECT_EQ(100, same_cell->value());
}

TEST(LabeledCounterDeathTest, DiesOnDecrement) {
  EXPECT_DEBUG_DEATH(
      { counter_with_labels->GetCell("DyingOp")->IncrementBy(-1); },
      "decrement");
}

auto* init_counter_without_labels = Counter<0>::New(
    "/tensorflow/test/init_counter_without_labels",
    "Counter without any labels to check if it is initialized as 0.");

TEST(UnlabeledCounterTest, InitializedWithZero) {
  EXPECT_EQ(0, init_counter_without_labels->GetCell()->value());
}

auto* counter_without_labels = Counter<0>::New(
    "/tensorflow/test/counter_without_labels", "Counter without any labels.");

TEST(UnlabeledCounterTest, GetCell) {
  auto* cell = counter_without_labels->GetCell();
  EXPECT_EQ(0, cell->value());

  cell->IncrementBy(42);
  EXPECT_EQ(42, cell->value());

  auto* same_cell = counter_without_labels->GetCell();
  EXPECT_EQ(42, same_cell->value());

  same_cell->IncrementBy(58);
  EXPECT_EQ(100, cell->value());
  EXPECT_EQ(100, same_cell->value());
}

auto* dead_counter_without_labels = Counter<0>::New(
    "/tensorflow/test/dead_counter_without_labels",
    "Counter without any labels which goes on to die on decrement.");

TEST(UnlabeledCounterDeathTest, DiesOnDecrement) {
  EXPECT_DEBUG_DEATH(
      { dead_counter_without_labels->GetCell()->IncrementBy(-1); },
      "decrement");
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
