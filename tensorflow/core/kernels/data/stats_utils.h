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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_STATS_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_STATS_UTILS_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace stats_utils {
extern const char kDelimiter[];
extern const char kExecutionTime[];
extern const char kThreadUtilization[];
extern const char kBufferSize[];
extern const char kBufferCapacity[];
extern const char kBufferUtilization[];
extern const char kFilteredElements[];
extern const char kDroppedElements[];
extern const char kFeaturesCount[];
extern const char kFeatureValuesCount[];
extern const char kExamplesCount[];

// Name for tf.data function execution time (in ns) histogram metrics.
string ExecutionTimeHistogramName(const string& prefix);

// Name for thread utilization (ratio of threads being used and maximum number
// of threads allocated) scalar metrics.
string ThreadUtilizationScalarName(const string& prefix);

// Name for buffer size scalar metrics.
string BufferSizeScalarName(const string& prefix);

// Name for buffer capacity (maximum allocated buffer size) scalar metrics.
string BufferCapacityScalarName(const string& prefix);

// Name for buffer utilization (ratio of buffer size and maximum allocated
// buffer size.) histogram metrics.
string BufferUtilizationHistogramName(const string& prefix);

// Name for filtered elements scalar metrics.
string FilterdElementsScalarName(const string& prefix);

// Name for dropped elements scalar mereics.
string DroppedElementsScalarName(const string& prefix);

// Name for features count histogram metrics.
string FeatureHistogramName(const string& prefix);

// Name for feature-values count histogram metrics.
string FeatureValueHistogramName(const string& prefix);

}  // namespace stats_utils
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_STATS_UTILS_H_
