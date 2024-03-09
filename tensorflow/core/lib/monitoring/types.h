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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_TYPES_H_
#define TENSORFLOW_CORE_LIB_MONITORING_TYPES_H_

#include <cmath>
#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tsl/lib/monitoring/types.h"

// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace monitoring {
using tsl::monitoring::PercentilePoint;
using tsl::monitoring::Percentiles;
using tsl::monitoring::UnitOfMeasure;

}  // namespace monitoring
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)
#endif  // TENSORFLOW_CORE_LIB_MONITORING_TYPES_H_
