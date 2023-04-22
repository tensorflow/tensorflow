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

#ifndef TENSORFLOW_COMPILER_XRT_XRT_METRICS_H_
#define TENSORFLOW_COMPILER_XRT_XRT_METRICS_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/lib/monitoring/percentile_sampler.h"

namespace tensorflow {
namespace xrt_metrics {

// Defines the singletons of the metrics populated by the XRT op framework.
// Single of a single XRT op there can be many device specific versions (CPU,
// GPU, TPU), and since the monitoring subsystem does not allow multiple
// registrations of the same metric name, we define them all in this file.
monitoring::PercentileSamplerCell* GetAllocateCell();
monitoring::PercentileSamplerCell* GetAllocateUninitializedCell();
monitoring::PercentileSamplerCell* GetAllocateFromTensorCell();
monitoring::PercentileSamplerCell* GetSubTupleCell();
monitoring::PercentileSamplerCell* GetMakeTupleCell();
monitoring::PercentileSamplerCell* GetReadLiteralCell();
monitoring::PercentileSamplerCell* GetReadToTensorCell();
monitoring::PercentileSamplerCell* GetWriteLiteralCell();
monitoring::PercentileSamplerCell* GetReleaseAllocationCell();
monitoring::PercentileSamplerCell* GetReleaseAllAllocationsCell();
monitoring::PercentileSamplerCell* GetCompactAllocationsCell();
monitoring::PercentileSamplerCell* GetCompileCell();
monitoring::PercentileSamplerCell* GetReleaseCompilationCell();
monitoring::PercentileSamplerCell* GetExecuteCell();
monitoring::PercentileSamplerCell* GetExecuteChainedCell();
monitoring::PercentileSamplerCell* GetMemoryCompactCell();
monitoring::PercentileSamplerCell* GetTryFreeMemoryCell();

}  // namespace xrt_metrics

xla::StatusOr<xrt::MetricsReport> CollectMetrics(
    const xrt::XRTMetricsCollect& metrics);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_METRICS_H_
