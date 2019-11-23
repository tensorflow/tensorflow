/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_PROFILER_INTERFACE_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_PROFILER_INTERFACE_H_

#include <memory>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace profiler {

enum class DeviceType {
  kUnspecified,
  kCpu,
  kGpu,
  kTpu,
};

struct ProfilerOptions {
  // DeviceType::kUnspecified: All registered device profiler will be enabled.
  // DeviceType::kCpu: only CPU will be profiled.
  // DeviceType::kGpu: only CPU/GPU will be profiled.
  // DeviceType::kTpu: only CPU/TPU will be profiled.
  DeviceType device_type = DeviceType::kUnspecified;

  // Inexpensive ops are not traced by default.
  int host_tracer_level = 2;
};

// Interface for tensorflow profiler plugins.
//
// ProfileSession calls each of these methods at most once per instance, and
// implementations can rely on that guarantee for simplicity.
//
// Thread-safety: Implementations are only required to be go/thread-compatible.
// ProfileSession is go/thread-safe and synchronizes access to ProfilerInterface
// instances.
class ProfilerInterface {
 public:
  virtual ~ProfilerInterface() = default;

  // Starts profiling.
  virtual Status Start() = 0;

  // Stops profiling.
  virtual Status Stop() = 0;

  // Saves collected profile data into step_stats_collector.
  // After this or the overload below are called once, subsequent calls might
  // return empty data.
  virtual Status CollectData(RunMetadata* run_metadata) = 0;

  // Saves collected profile data into XSpace.
  // After this or the overload above are called once, subsequent calls might
  // return empty data.
  virtual Status CollectData(XSpace* space) = 0;

  // Which device this ProfilerInterface is used for.
  virtual DeviceType GetDeviceType() = 0;
};

}  // namespace profiler

using ProfilerFactory = std::unique_ptr<profiler::ProfilerInterface> (*)(
    const profiler::ProfilerOptions&);

void RegisterProfilerFactory(ProfilerFactory factory);

void CreateProfilers(
    const profiler::ProfilerOptions& options,
    std::vector<std::unique_ptr<profiler::ProfilerInterface>>* result);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_PROFILER_INTERFACE_H_
