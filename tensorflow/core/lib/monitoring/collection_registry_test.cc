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

#include "tensorflow/core/lib/monitoring/collection_registry.h"

#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

void EmptyCollectionFunction(MetricCollectorGetter getter) {}

TEST(CollectionRegistryTest, RegistrationUnregistration) {
  auto* collection_registry = CollectionRegistry::Default();
  const MetricDef<MetricKind::kCumulative, int64, 0> metric_def0(
      "/tensorflow/metric0", "An example metric with no labels.");
  const MetricDef<MetricKind::kGauge, double, 1> metric_def1(
      "/tensorflow/metric1", "An example metric with one label.", "LabelName");

  {
    // Enclosed in a scope so that we unregister before the stack variables
    // above are destroyed.

    std::unique_ptr<CollectionRegistry::RegistrationHandle> handle0 =
        collection_registry->Register(&metric_def0, EmptyCollectionFunction);
    std::unique_ptr<CollectionRegistry::RegistrationHandle> handle1 =
        collection_registry->Register(&metric_def1, EmptyCollectionFunction);

    handle0.reset();

    // Able to register again because it was unregistered earlier.
    handle0 =
        collection_registry->Register(&metric_def0, EmptyCollectionFunction);
  }
}

TEST(CollectionRegistryDeathTest, DuplicateRegistration) {
  auto* collection_registry = CollectionRegistry::Default();
  const MetricDef<MetricKind::kCumulative, int64, 0> metric_def(
      "/tensorflow/metric", "An example metric with no labels.");

  auto handle =
      collection_registry->Register(&metric_def, EmptyCollectionFunction);
  EXPECT_DEATH(
      {
        auto duplicate_handle =
            collection_registry->Register(&metric_def, EmptyCollectionFunction);
      },
      "/tensorflow/metric");
}

auto* counter_with_labels =
    Counter<2>::New({"/tensorflow/test/counter_with_labels",
                     "Counter with one label.", "MyLabel0", "MyLabel1"});
auto* counter_without_labels = Counter<0>::New(
    {"/tensorflow/test/counter_without_labels", "Counter without any labels."});
TEST(CollectMetricsTest, Counter) {
  counter_with_labels->GetCell("Label00", "Label10")->IncrementBy(42);
  counter_with_labels->GetCell("Label01", "Label11")->IncrementBy(58);
  counter_without_labels->GetCell()->IncrementBy(7);

  auto* collection_registry = CollectionRegistry::Default();
  const std::unique_ptr<CollectedMetrics> collected_metrics =
      collection_registry->CollectMetrics();

  ASSERT_EQ(2, collected_metrics->metric_descriptor_map.size());

  const MetricDescriptor& ld = *collected_metrics->metric_descriptor_map.at(
      "/tensorflow/test/counter_with_labels");
  EXPECT_EQ("/tensorflow/test/counter_with_labels", ld.name);
  EXPECT_EQ("Counter with one label.", ld.description);
  ASSERT_EQ(2, ld.label_names.size());
  EXPECT_EQ("MyLabel0", ld.label_names[0]);
  EXPECT_EQ("MyLabel1", ld.label_names[1]);
  EXPECT_EQ(MetricKind::kCumulative, ld.metric_kind);
  EXPECT_EQ(ValueType::kInt64, ld.value_type);

  const MetricDescriptor& ud = *collected_metrics->metric_descriptor_map.at(
      "/tensorflow/test/counter_without_labels");
  EXPECT_EQ("/tensorflow/test/counter_without_labels", ud.name);
  EXPECT_EQ("Counter without any labels.", ud.description);
  ASSERT_EQ(0, ud.label_names.size());
  EXPECT_EQ(MetricKind::kCumulative, ud.metric_kind);
  EXPECT_EQ(ValueType::kInt64, ud.value_type);

  ASSERT_EQ(2, collected_metrics->point_set_map.size());

  const PointSet& lps = *collected_metrics->point_set_map.at(
      "/tensorflow/test/counter_with_labels");
  EXPECT_EQ("/tensorflow/test/counter_with_labels", lps.metric_name);
  ASSERT_EQ(2, lps.points.size());
  ASSERT_EQ(2, lps.points[0]->labels.size());
  EXPECT_EQ("MyLabel0", lps.points[0]->labels[0].name);
  EXPECT_EQ("Label00", lps.points[0]->labels[0].value);
  EXPECT_EQ("MyLabel1", lps.points[0]->labels[1].name);
  EXPECT_EQ("Label10", lps.points[0]->labels[1].value);
  EXPECT_EQ(ValueType::kInt64, lps.points[0]->value_type);
  EXPECT_EQ(42, lps.points[0]->int64_value);
  ASSERT_EQ(2, lps.points[1]->labels.size());
  EXPECT_EQ("MyLabel0", lps.points[1]->labels[0].name);
  EXPECT_EQ("Label01", lps.points[1]->labels[0].value);
  EXPECT_EQ("MyLabel1", lps.points[1]->labels[1].name);
  EXPECT_EQ("Label11", lps.points[1]->labels[1].value);
  EXPECT_EQ(ValueType::kInt64, lps.points[1]->value_type);
  EXPECT_EQ(58, lps.points[1]->int64_value);

  const PointSet& ups = *collected_metrics->point_set_map.at(
      "/tensorflow/test/counter_without_labels");
  EXPECT_EQ("/tensorflow/test/counter_without_labels", ups.metric_name);
  ASSERT_EQ(1, ups.points.size());
  EXPECT_EQ(0, ups.points[0]->labels.size());
  EXPECT_EQ(ValueType::kInt64, ups.points[0]->value_type);
  EXPECT_EQ(7, ups.points[0]->int64_value);
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
