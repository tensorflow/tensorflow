/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_INL_H_
#define TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_INL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/test_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/tsl/lib/monitoring/cell_reader-inl.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace monitoring {
namespace testing {
namespace internal {
using tsl::monitoring::testing::internal::CollectMetrics;
using tsl::monitoring::testing::internal::GetDelta;
using tsl::monitoring::testing::internal::GetLatestPoint;
using tsl::monitoring::testing::internal::GetLatestValueOrDefault;
using tsl::monitoring::testing::internal::GetMetricKind;
using tsl::monitoring::testing::internal::GetPoints;
using tsl::monitoring::testing::internal::GetValue;
}  // namespace internal
}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)
#endif  // TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_INL_H_
