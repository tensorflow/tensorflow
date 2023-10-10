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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
#define TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_

#include "tsl/lib/monitoring/collection_registry.h"
// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on
// We use a null implementation for mobile platforms.
#ifdef IS_MOBILE_PLATFORM

#include <functional>
#include <map>
#include <memory>

#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace monitoring {
using tsl::monitoring::CollectionRegistry;
using tsl::monitoring::MetricCollector;
using tsl::monitoring::MetricCollectorGetter;
}  // namespace monitoring
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)
#else  // !defined(IS_MOBILE_PLATFORM)

#include <functional>
#include <map>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::monitoring::CollectionRegistry;
using tsl::monitoring::Exporter;
using tsl::monitoring::MetricCollector;
using tsl::monitoring::MetricCollectorGetter;
using tsl::monitoring::exporter_registration::ExporterRegistration;
using tsl::monitoring::internal::Collector;
namespace test_util {
class CollectionRegistryTestAccess;
}  // namespace test_util
// NOLINTEND(misc-unused-using-decls)
}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM

#endif  // TENSORFLOW_CORE_LIB_MONITORING_COLLECTION_REGISTRY_H_
