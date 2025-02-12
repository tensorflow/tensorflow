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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_PERCENTILE_SAMPLER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_PERCENTILE_SAMPLER_H_

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
#include "xla/tsl/lib/monitoring/percentile_sampler.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/macros.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace monitoring {

using tsl::monitoring::PercentileSampler;
using tsl::monitoring::PercentileSamplerCell;

}  // namespace monitoring
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)
#else  // IS_MOBILE_PLATFORM

#include <cmath>
#include <map>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::monitoring::PercentileSampler;
using tsl::monitoring::PercentileSamplerCell;
// NOLINTEND(misc-unused-using-decls)
}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_CORE_LIB_MONITORING_PERCENTILE_SAMPLER_H_
