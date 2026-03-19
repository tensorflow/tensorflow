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
#ifndef TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/lib/monitoring/cell_reader.h"
#include "tensorflow/core/lib/monitoring/cell_reader-inl.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace monitoring {
namespace testing {
using tsl::monitoring::testing::CellReader;
}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)
#endif  // TENSORFLOW_CORE_LIB_MONITORING_CELL_READER_H_
