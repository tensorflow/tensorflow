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

#include "tensorflow/core/common_runtime/no_op_cost_measurement.h"

#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/cost_constants.h"

namespace tensorflow {

absl::Duration NoOpCostMeasurement::GetTotalCost() { return absl::Duration(); }

absl::string_view NoOpCostMeasurement::GetCostType() const {
  return kNoOpCostName;
}

REGISTER_COST_MEASUREMENT(kNoOpCostName, NoOpCostMeasurement);

}  // namespace tensorflow
