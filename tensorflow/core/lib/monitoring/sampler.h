/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_

#include "xla/tsl/lib/monitoring/sampler.h"
#ifdef IS_MOBILE_PLATFORM

#include <memory>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#else  // IS_MOBILE_PLATFORM

#include <float.h>

#include <map>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#endif
// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace monitoring {

using tsl::monitoring::Buckets;
using tsl::monitoring::Sampler;
using tsl::monitoring::SamplerCell;
}  // namespace monitoring
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)
#endif  // TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
