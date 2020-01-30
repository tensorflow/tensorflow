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

#include "tensorflow/compiler/xrt/xrt_metrics.h"

#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace {

static const size_t kMaxSamples = 1024;

std::vector<double> GetDefaultPercentiles() {
  return {25.0, 50.0, 80.0, 90.0, 95.0, 99.0};
}

bool IsSelectedMetric(const xrt::XRTMetricsCollect& metrics,
                      const string& name) {
  if (metrics.metrics_regex_size() == 0) {
    return true;
  }
  for (auto& metric_regex : metrics.metrics_regex()) {
    if (RE2::FullMatch(name, metric_regex)) {
      return true;
    }
  }
  return false;
}

Status AddMetrics(xrt::MetricsReport* report,
                  const monitoring::PointSet& point_set) {
  for (auto& point : point_set.points) {
    xrt::MetricValues* metrics = report->add_metrics();
    metrics->set_name(point_set.metric_name);
    if (point->value_type == monitoring::ValueType::kPercentiles) {
      xrt::Percentiles* percentiles = metrics->mutable_percentiles_value();
      percentiles->set_start_nstime(point->percentiles_value.start_nstime);
      percentiles->set_end_nstime(point->percentiles_value.end_nstime);
      percentiles->set_min_value(point->percentiles_value.min_value);
      percentiles->set_max_value(point->percentiles_value.max_value);
      percentiles->set_mean(point->percentiles_value.mean);
      percentiles->set_stddev(point->percentiles_value.stddev);
      percentiles->set_num_samples(point->percentiles_value.num_samples);
      percentiles->set_total_samples(point->percentiles_value.total_samples);
      percentiles->set_accumulator(point->percentiles_value.accumulator);
      for (auto& pct_point : point->percentiles_value.points) {
        xrt::Percentiles::Point* xpoint = percentiles->add_points();
        xpoint->set_percentile(pct_point.percentile);
        xpoint->set_value(pct_point.value);
      }
    } else if (point->value_type == monitoring::ValueType::kInt64) {
      metrics->set_int64_value(point->int64_value);
    }
  }
  return Status::OK();
}

}  // namespace

namespace xrt_metrics {

monitoring::PercentileSamplerCell* GetAllocateCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/allocate", "Tracks XRTAllocate times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetAllocateUninitializedCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/allocate_uninitialized",
           "Tracks XRTAllocateUninitialized times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetAllocateFromTensorCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/allocate_from_tensor",
           "Tracks XRTAllocateFromTensor times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetSubTupleCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/sub_tuple", "Tracks XRTSubTuple times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetMakeTupleCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/make_tuple", "Tracks XRTMakeTuple times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReadLiteralCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/read_literal", "Tracks XRTReadLiteral times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReadToTensorCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/read_tensor", "Tracks XRTReadToTensor times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetWriteLiteralCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/write_literal", "Tracks XRTWriteLiteral times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReleaseAllocationCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/release_allocation",
           "Tracks XRTReleaseAllocation times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReleaseAllAllocationsCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/release_all_allocations",
           "Tracks XRTReleaseAllAllocations times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetCompactAllocationsCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/compact_allocations",
           "Tracks XRTCompactAllocations times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetCompileCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/compile", "Tracks XRTCompile times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetReleaseCompilationCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/release_compilation",
           "Tracks XRTReleaseCompilationRef times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetExecuteCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/execute", "Tracks XRTExecute times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetExecuteChainedCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/ops/execute_chained",
           "Tracks XRTExecuteChained times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetMemoryCompactCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/memory_manager/compaction",
           "Tracks XRT memory manager memory compaction times"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

monitoring::PercentileSamplerCell* GetTryFreeMemoryCell() {
  static monitoring::PercentileSamplerCell* cell =
      monitoring::PercentileSampler<0>::New(
          {"/tensorflow/xrt/memory_manager/try_free_memory",
           "Tracks XRT memory manager times in trying to "
           "free memory by swpping device memory to host memory"},
          GetDefaultPercentiles(), kMaxSamples)
          ->GetCell();
  return cell;
}

}  // namespace xrt_metrics

xla::StatusOr<xrt::MetricsReport> CollectMetrics(
    const xrt::XRTMetricsCollect& metrics) {
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  options.collect_metric_descriptors = false;
  auto collected_metrics = collection_registry->CollectMetrics(options);
  xrt::MetricsReport report;
  for (auto& name_pointset : collected_metrics->point_set_map) {
    if (IsSelectedMetric(metrics, name_pointset.first)) {
      TF_RETURN_IF_ERROR(AddMetrics(&report, *name_pointset.second));
    }
  }
  return std::move(report);
}

}  // namespace tensorflow
