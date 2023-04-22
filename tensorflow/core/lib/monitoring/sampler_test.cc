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

#include "tensorflow/core/lib/monitoring/sampler.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

using histogram::Histogram;

void EqHistograms(const Histogram& expected,
                  const HistogramProto& actual_proto) {
  Histogram actual;
  ASSERT_TRUE(actual.DecodeFromProto(actual_proto));

  EXPECT_EQ(expected.ToString(), actual.ToString());
}

auto* sampler_with_labels =
    Sampler<1>::New({"/tensorflow/test/sampler_with_labels",
                     "Sampler with one label.", "MyLabel"},
                    Buckets::Explicit({10.0, 20.0}));

TEST(LabeledSamplerTest, InitializedEmpty) {
  Histogram empty;
  EqHistograms(empty, sampler_with_labels->GetCell("Empty")->value());
}

TEST(LabeledSamplerTest, ExplicitBucketBoundaries) {
  // Sampler automatically adds DBL_MAX to the list of buckets.
  Histogram expected({10.0, 20.0, DBL_MAX});
  auto* cell = sampler_with_labels->GetCell("BucketBoundaries");
  sampler_with_labels->GetCell("AddedToCheckPreviousCellValidity");
  cell->Add(-1.0);
  expected.Add(-1.0);
  cell->Add(10.0);
  expected.Add(10.0);
  cell->Add(20.0);
  expected.Add(20.0);
  cell->Add(31.0);
  expected.Add(31.0);

  EqHistograms(expected, cell->value());
}

auto* init_sampler_without_labels =
    Sampler<0>::New({"/tensorflow/test/init_sampler_without_labels",
                     "Sampler without labels initialized as empty."},
                    Buckets::Explicit(std::vector<double>{1.5, 2.8}));

TEST(UnlabeledSamplerTest, InitializedEmpty) {
  Histogram empty;
  EqHistograms(empty, init_sampler_without_labels->GetCell()->value());
}

auto* sampler_without_labels =
    Sampler<0>::New({"/tensorflow/test/sampler_without_labels",
                     "Sampler without labels initialized as empty."},
                    Buckets::Explicit({1.5, 2.8}));

TEST(UnlabeledSamplerTest, ExplicitBucketBoundaries) {
  // Sampler automatically adds DBL_MAX to the list of buckets.
  Histogram expected({1.5, 2.8, DBL_MAX});
  auto* cell = sampler_without_labels->GetCell();
  cell->Add(-1.0);
  expected.Add(-1.0);
  cell->Add(2.0);
  expected.Add(2.0);
  cell->Add(31.0);
  expected.Add(31.0);

  EqHistograms(expected, cell->value());
}

auto* sampler_with_exponential =
    Sampler<1>::New({"/tensorflow/test/sampler_with_exponential",
                     "Sampler with exponential buckets.", "MyLabel"},
                    // So limits are {1, 2, 4}.
                    Buckets::Exponential(1, 2, 3));

TEST(ExponentialSamplerTest, ExponentialBucketBoundaries) {
  // Sampler automatically adds DBL_MAX to the list of buckets.
  Histogram expected({1.0, 2.0, 4.0, DBL_MAX});
  auto* cell = sampler_with_exponential->GetCell("BucketBoundaries");
  sampler_with_exponential->GetCell("AddedToCheckPreviousCellValidity");
  cell->Add(-1.0);
  expected.Add(-1.0);
  cell->Add(0.5);
  expected.Add(0.5);
  cell->Add(1.001);
  expected.Add(1.001);
  cell->Add(3.999);
  expected.Add(3.999);
  cell->Add(6.0);
  expected.Add(6.0);

  EqHistograms(expected, cell->value());
}

TEST(ExplicitSamplerTest, SameName) {
  auto* same_sampler = Sampler<1>::New({"/tensorflow/test/sampler_with_labels",
                                        "Sampler with one label.", "MyLabel"},
                                       Buckets::Explicit({10.0, 20.0}));
  EXPECT_TRUE(sampler_with_labels->GetStatus().ok());
  EXPECT_FALSE(same_sampler->GetStatus().ok());
  delete same_sampler;
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
