/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/monitoring/percentile_sampler.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

auto* pctsampler_with_labels = PercentileSampler<1>::New(
    {"/tensorflow/test/percentile_sampler_with_labels",
     "Percentile sampler with one label.", "MyLabel"},
    {25.0, 50.0, 90.0, 99.0}, 1024, UnitOfMeasure::kNumber);
auto* pctsampler_without_labels = PercentileSampler<0>::New(
    {"/tensorflow/test/percentile_sampler_without_labels",
     "Percentile sampler without labels initialized as empty."},
    {25.0, 50.0, 90.0, 99.0}, 1024, UnitOfMeasure::kNumber);

TEST(LabeledPercentileSamplerTest, FixedPercentilesValues) {
  auto* cell = pctsampler_with_labels->GetCell("MyLabel");
  cell->Add(10.0);
  cell->Add(4.0);
  cell->Add(1.0);
  cell->Add(0.6);

  auto value = cell->value();
  EXPECT_EQ(value.min_value, 0.6);
  EXPECT_EQ(value.max_value, 10.0);
  EXPECT_EQ(value.num_samples, 4);

  EXPECT_EQ(value.points[0].value, 1.0);
  EXPECT_EQ(value.points[1].value, 4.0);
  EXPECT_EQ(value.points[2].value, 10.0);
  EXPECT_EQ(value.points[3].value, 10.0);
}

TEST(UnlabeledPercentileSamplerTest, FixedPercentilesValues) {
  auto* cell = pctsampler_without_labels->GetCell();
  cell->Add(10.0);
  cell->Add(4.0);
  cell->Add(1.0);
  cell->Add(0.6);

  auto value = cell->value();
  EXPECT_EQ(value.min_value, 0.6);
  EXPECT_EQ(value.max_value, 10.0);
  EXPECT_EQ(value.num_samples, 4);

  EXPECT_EQ(value.points[0].value, 1.0);
  EXPECT_EQ(value.points[1].value, 4.0);
  EXPECT_EQ(value.points[2].value, 10.0);
  EXPECT_EQ(value.points[3].value, 10.0);
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
