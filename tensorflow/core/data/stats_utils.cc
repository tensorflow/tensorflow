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
#include "tensorflow/core/data/stats_utils.h"

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

std::string ExecutionTimeHistogramName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kExecutionTime);
}

std::string ThreadUtilizationScalarName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kThreadUtilization);
}

std::string BufferSizeScalarName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kBufferSize);
}

std::string BufferCapacityScalarName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kBufferCapacity);
}

std::string BufferUtilizationHistogramName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kBufferUtilization);
}

std::string FilterdElementsScalarName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kFilteredElements);
}

std::string DroppedElementsScalarName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kDroppedElements);
}

std::string FeatureHistogramName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kFeaturesCount);
}

std::string FeatureValueHistogramName(const std::string& prefix) {
  return absl::StrCat(prefix, kDelimiter, kFeatureValuesCount);
}

}  // namespace stats_utils
}  // namespace data
}  // namespace tensorflow
