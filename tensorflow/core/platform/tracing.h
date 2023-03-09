/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_TRACING_H_
#define TENSORFLOW_CORE_PLATFORM_TRACING_H_

// Tracing interface

#include <array>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/platform/tracing.h"

namespace tensorflow {
namespace tracing {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::tracing::EventCategory;
using tsl::tracing::EventCollector;
using tsl::tracing::GetArgForName;
using tsl::tracing::GetEventCategoryName;
using tsl::tracing::GetEventCollector;
using tsl::tracing::GetLogDir;
using tsl::tracing::GetNumEventCategories;
using tsl::tracing::GetUniqueArg;
using tsl::tracing::RecordEvent;
using tsl::tracing::ScopedRegion;
using tsl::tracing::SetEventCollector;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tracing
}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/tsl/platform/google/tracing_impl.h"
#else
#include "tensorflow/tsl/platform/default/tracing_impl.h"
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_TRACING_H_
