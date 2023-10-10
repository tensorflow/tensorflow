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

#include "tensorflow/core/common_runtime/cost_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

namespace {

// Gets the types of CostMeasurement from env.
std::vector<std::string> GetCostMeasurementTypes() {
  const char* types = std::getenv("TF_COST_MEASUREMENT_TYPE");
  if (types == nullptr) return {};
  return str_util::Split(types, " ,");
}

// Gets the type of RequestCostAccessor from env.
const char* GetRequestCostAccessorType() {
  static const char* accessor = std::getenv("TF_REQUEST_COST_ACCESSOR_TYPE");
  return accessor;
}

}  // namespace

std::vector<std::unique_ptr<CostMeasurement>> CreateCostMeasurements(
    const CostMeasurement::Context& context) {
  static const std::vector<std::string>& types =
      *new std::vector<std::string>(GetCostMeasurementTypes());

  std::vector<std::unique_ptr<CostMeasurement>> measurements;
  for (const auto& type : types) {
    std::unique_ptr<CostMeasurement> measurement =
        CostMeasurementRegistry::CreateByNameOrNull(type, context);
    if (measurement != nullptr) {
      measurements.push_back(std::move(measurement));
    }
  }
  return measurements;
}

std::unique_ptr<RequestCostAccessor> CreateRequestCostAccessor() {
  const char* request_cost_accessor_type = GetRequestCostAccessorType();
  return request_cost_accessor_type
             ? RequestCostAccessorRegistry::CreateByNameOrNull(
                   request_cost_accessor_type)
             : nullptr;
}

}  // namespace tensorflow
