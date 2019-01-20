/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/monitoring/gauge.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

auto* gauge_with_labels = Gauge<int64, 1>::New(
    "/tensorflow/test/gauge_with_labels", "Gauge with one label.", "MyLabel");

TEST(LabeledGaugeTest, InitializedWithZero) {
  EXPECT_EQ(0, gauge_with_labels->GetCell("Empty")->value());
}

TEST(LabeledGaugeTest, GetCell) {
  auto* cell = gauge_with_labels->GetCell("GetCellOp");
  EXPECT_EQ(0, cell->value());

  cell->Set(1);
  EXPECT_EQ(1, cell->value());

  auto* same_cell = gauge_with_labels->GetCell("GetCellOp");
  EXPECT_EQ(1, same_cell->value());

  same_cell->Set(10);
  EXPECT_EQ(10, cell->value());
  EXPECT_EQ(10, same_cell->value());
}

auto* gauge_without_labels = Gauge<int64, 0>::New(
    "/tensorflow/test/gauge_without_labels", "Gauge without any labels.");

TEST(UnlabeledGaugeTest, InitializedWithZero) {
  EXPECT_EQ(0, gauge_without_labels->GetCell()->value());
}

TEST(UnlabeledGaugeTest, GetCell) {
  auto* cell = gauge_without_labels->GetCell();
  EXPECT_EQ(0, cell->value());

  cell->Set(1);
  EXPECT_EQ(1, cell->value());

  auto* same_cell = gauge_without_labels->GetCell();
  EXPECT_EQ(1, same_cell->value());

  same_cell->Set(10);
  EXPECT_EQ(10, cell->value());
  EXPECT_EQ(10, same_cell->value());
}

auto* string_gauge = Gauge<string, 0>::New("/tensorflow/test/string_gauge",
                                           "Gauge of string value.");

TEST(GaugeOfStringValue, InitializedWithEmptyString) {
  EXPECT_EQ("", string_gauge->GetCell()->value());
}

TEST(GaugeOfStringValue, GetCell) {
  auto* cell = string_gauge->GetCell();
  EXPECT_EQ("", cell->value());

  cell->Set("foo");
  EXPECT_EQ("foo", cell->value());

  auto* same_cell = string_gauge->GetCell();
  EXPECT_EQ("foo", cell->value());

  same_cell->Set("bar");
  EXPECT_EQ("bar", cell->value());
  EXPECT_EQ("bar", same_cell->value());
}

auto* bool_gauge =
    Gauge<bool, 0>::New("/tensorflow/test/bool_gauge", "Gauge of bool value.");

TEST(GaugeOfBoolValue, InitializedWithFalseValue) {
  EXPECT_EQ(false, bool_gauge->GetCell()->value());
}

TEST(GaugeOfBoolValue, GetCell) {
  auto* cell = bool_gauge->GetCell();
  EXPECT_EQ(false, cell->value());

  cell->Set(true);
  EXPECT_EQ(true, cell->value());

  auto* same_cell = bool_gauge->GetCell();
  EXPECT_EQ(true, cell->value());

  same_cell->Set(false);
  EXPECT_EQ(false, cell->value());
  EXPECT_EQ(false, same_cell->value());
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
