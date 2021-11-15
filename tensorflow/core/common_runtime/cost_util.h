/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COST_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COST_UTIL_H_

#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/request_cost_accessor.h"

namespace tensorflow {

// Creates instances of CostMeasurement. The types to create are determined by
// env.
std::vector<std::unique_ptr<CostMeasurement>> CreateCostMeasurements();

// Creates an instance of RequestCostAccessor. The type to create is determined
// by env. Returns nullptr if the type is not specified in env, or the type of
// CostMeasurement is unregistered..
std::unique_ptr<RequestCostAccessor> CreateRequestCostAccessor();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COST_UTIL_H_
