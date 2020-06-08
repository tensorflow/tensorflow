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
#include "tensorflow/core/kernels/data/stats_utils.h"

#include "absl/base/attributes.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace data {
namespace stats_utils {

ABSL_CONST_INIT const char kDelimiter[] = "::";
ABSL_CONST_INIT const char kExecutionTime[] = "execution_time";
ABSL_CONST_INIT const char kThreadUtilization[] = "thread_utilization";
ABSL_CONST_INIT const char kBufferSize[] = "buffer_size";
ABSL_CONST_INIT const char kBufferCapacity[] = "buffer_capacity";
ABSL_CONST_INIT const char kBufferUtilization[] = "buffer_utilization";
ABSL_CONST_INIT const char kFilteredElements[] = "filtered_elements";
ABSL_CONST_INIT const char kDroppedElements[] = "dropped_elements";
ABSL_CONST_INIT const char kFeaturesCount[] = "features_count";
ABSL_CONST_INIT const char kFeatureValuesCount[] = "feature_values_count";
ABSL_CONST_INIT const char kExamplesCount[] = "examples_count";

string ExecutionTimeHistogramName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kExecutionTime);
}

string ThreadUtilizationScalarName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kThreadUtilization);
}

string BufferSizeScalarName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kBufferSize);
}

string BufferCapacityScalarName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kBufferCapacity);
}

string BufferUtilizationHistogramName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kBufferUtilization);
}

string FilterdElementsScalarName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kFilteredElements);
}

string DroppedElementsScalarName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kDroppedElements);
}

string FeatureHistogramName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kFeaturesCount);
}

string FeatureValueHistogramName(const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, kFeatureValuesCount);
}

}  // namespace stats_utils
}  // namespace data
}  // namespace tensorflow
