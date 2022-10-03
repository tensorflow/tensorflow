/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_METRICS_H_
#define TENSORFLOW_CORE_FRAMEWORK_METRICS_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/tsl/framework/metrics.h"

namespace tensorflow {
namespace metrics {

using tsl::metrics::GetGraphOptimizationCounter;
using tsl::metrics::GetTFDataBytesConsumedCounter;
using tsl::metrics::GetTFDataBytesProducedCounter;
using tsl::metrics::GetTFDataBytesReadCounter;
using tsl::metrics::GetTFDataElementsCounter;
using tsl::metrics::GetTFDataModelGauge;
using tsl::metrics::IncrementTestCounter;
using tsl::metrics::RecordGraphInputTensors;
using tsl::metrics::RecordGraphOutputTensors;
using tsl::metrics::RecordParseDenseFeature;
using tsl::metrics::RecordParseRaggedFeature;
using tsl::metrics::RecordParseSparseFeature;
using tsl::metrics::RecordTFDataAutoShard;
using tsl::metrics::RecordTFDataAutoShardRewriteBatchSize;
using tsl::metrics::RecordTFDataAutotune;
using tsl::metrics::RecordTFDataAutotuneMaxBufferBudgetRatio;
using tsl::metrics::RecordTFDataAutotuneStoppingCriteria;
using tsl::metrics::RecordTFDataAutotuneUsedRamBudgetRatio;
using tsl::metrics::RecordTFDataBytesFetched;
using tsl::metrics::RecordTFDataExperiment;
using tsl::metrics::RecordTFDataFilename;
using tsl::metrics::RecordTFDataFingerprint;
using tsl::metrics::RecordTFDataGetNextDuration;
using tsl::metrics::RecordTFDataIteratorBusy;
using tsl::metrics::RecordTFDataIteratorGap;
using tsl::metrics::RecordTFDataIteratorLifetime;
using tsl::metrics::RecordTFDataOptimization;
using tsl::metrics::RecordTFDataServiceClientIterators;
using tsl::metrics::RecordTFDataServiceCrossTrainerCacheQuery;
using tsl::metrics::RecordTFDataServiceCrossTrainerCacheSizeBytes;
using tsl::metrics::RecordTFDataServiceJobsCreated;
using tsl::metrics::RecordTFDataServiceWorkerCreated;
using tsl::metrics::RecordTPUXlaSpmdCoresPerReplica;
using tsl::metrics::RecordUnusedOutput;
using tsl::metrics::ScopedCounter;
using tsl::metrics::TestCounter;
using tsl::metrics::TestDelta;
using tsl::metrics::UpdateBfcAllocatorDelayTime;
using tsl::metrics::UpdateEagerClientErrorCounter;
using tsl::metrics::UpdateFunctionGraphOptimizationTime;
using tsl::metrics::UpdateGraphBuildTime;
using tsl::metrics::UpdateGraphExecTime;
using tsl::metrics::UpdateGraphPendingQueueLength;
using tsl::metrics::UpdateTfMlirBridgeFirstPhaseCounter;
using tsl::metrics::UpdateTfMlirBridgeGraphAnalysisPerOp;
using tsl::metrics::UpdateTpuErrorCounter;
using tsl::metrics::UpdateTpuVariableDistributionTime;
using tsl::metrics::UpdateXlaCompilationTime;

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_METRICS_H_
