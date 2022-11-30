/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_

#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tensorflow/tsl/profiler/utils/group_events.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::CheckLoopOp;                       // NOLINT
using tsl::profiler::ContextGroup;                      // NOLINT
using tsl::profiler::ContextGroupMap;                   // NOLINT
using tsl::profiler::CreateInterThreadConnectInfoList;  // NOLINT
using tsl::profiler::EventForest;                       // NOLINT
using tsl::profiler::EventList;                         // NOLINT
using tsl::profiler::EventNode;                         // NOLINT
using tsl::profiler::EventNodeMap;                      // NOLINT
using tsl::profiler::GroupMetadata;                     // NOLINT
using tsl::profiler::GroupMetadataMap;                  // NOLINT
using tsl::profiler::GroupTfEvents;                     // NOLINT
using tsl::profiler::InterThreadConnectInfo;            // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_GROUP_EVENTS_H_
