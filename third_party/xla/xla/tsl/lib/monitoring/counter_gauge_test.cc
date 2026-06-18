/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/monitoring/counter_gauge.h"

#include <gtest/gtest.h>

namespace tsl::monitoring {
namespace {

auto* counter_gauge_with_labels =
    CounterGauge<1>::New("/tensorflow/test/counter_gauge_with_labels",
                         "CounterGauge with one label.", "MyLabel");

TEST(LabeledCounterGaugeTest, InitializedWithZero) {
  EXPECT_EQ(0, counter_gauge_with_labels->GetCell("Empty")->value());
}

TEST(LabeledCounterGaugeTest, GetCell) {
  auto* cell = counter_gauge_with_labels->GetCell("GetCellOp");
  EXPECT_EQ(0, cell->value());

  cell->IncrementBy(-42);
  EXPECT_EQ(-42, cell->value());

  auto* same_cell = counter_gauge_with_labels->GetCell("GetCellOp");
  EXPECT_EQ(-42, same_cell->value());

  same_cell->IncrementBy(42);
  EXPECT_EQ(0, cell->value());
  EXPECT_EQ(0, same_cell->value());
}

TEST(LabeledCounterGaugeTest, IncrementAndDecrement) {
  auto* cell = counter_gauge_with_labels->GetCell("IncrementAndDecrementOp");
  cell->Increment();
  EXPECT_EQ(1, cell->value());
  cell->Increment();
  EXPECT_EQ(2, cell->value());
  cell->Decrement();
  EXPECT_EQ(1, cell->value());
  cell->Decrement();
  EXPECT_EQ(0, cell->value());
}

TEST(LabeledCounterGaugeTest, SameName) {
  auto* same_counter =
      CounterGauge<1>::New("/tensorflow/test/counter_gauge_with_labels",
                           "Counter with one label.", "MyLabel");
  EXPECT_TRUE(counter_gauge_with_labels->GetStatus().ok());
  EXPECT_TRUE(same_counter->GetStatus().ok());
  delete same_counter;
}

auto* init_counter_gauge_without_labels = CounterGauge<0>::New(
    "/tensorflow/test/init_counter_gauge_without_labels",
    "Counter without any labels to check if it is initialized as 0.");

TEST(UnlabeledCounterGaugeTest, InitializedWithZero) {
  EXPECT_EQ(0, init_counter_gauge_without_labels->GetCell()->value());
}

auto* counter_gauge_without_labels =
    CounterGauge<0>::New("/tensorflow/test/counter_gauge_without_labels",
                         "Counter without any labels.");

TEST(UnlabeledCounterGaugeTest, GetCell) {
  auto* cell = counter_gauge_without_labels->GetCell();
  EXPECT_EQ(0, cell->value());

  cell->IncrementBy(42);
  EXPECT_EQ(42, cell->value());

  auto* same_cell = counter_gauge_without_labels->GetCell();
  EXPECT_EQ(42, same_cell->value());

  same_cell->IncrementBy(58);
  EXPECT_EQ(100, cell->value());
  EXPECT_EQ(100, same_cell->value());

  cell->IncrementBy(-100);
  EXPECT_EQ(0, cell->value());
  EXPECT_EQ(0, same_cell->value());
}

TEST(UnlabeledCounterGaugeTest, IncrementAndDecrement) {
  auto* cell = counter_gauge_without_labels->GetCell();
  cell->Increment();
  EXPECT_EQ(1, cell->value());
  cell->Increment();
  EXPECT_EQ(2, cell->value());
  cell->Decrement();
  EXPECT_EQ(1, cell->value());
  cell->Decrement();
  EXPECT_EQ(0, cell->value());
}

}  // namespace
}  // namespace tsl::monitoring
