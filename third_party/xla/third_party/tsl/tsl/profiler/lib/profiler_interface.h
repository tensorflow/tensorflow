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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_

#include <any>
#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

struct ConsumeResult {
  std::any data;
  size_t estimated_size_bytes = 0;
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
  virtual absl::Status Start() = 0;

  // Stops profiling.
  virtual absl::Status Stop() = 0;

  // Saves collected profile data into XSpace.
  virtual absl::Status CollectData(tensorflow::profiler::XSpace* space) = 0;

  // Consumes collected profile data without stopping the profiler.
  virtual absl::StatusOr<ConsumeResult> Consume() {
    return absl::UnimplementedError("Consume not implemented");
  }

  // Serializes consumed profile data into XSpace.
  virtual absl::Status Serialize(std::any data,
                                 tensorflow::profiler::XSpace* space) {
    return absl::UnimplementedError("Serialize not implemented");
  }
};

// MultiPassProfilerInterface manages multi-pass profiling plugins.
// Implementations plan the passes (e.g. counter partitioning), configure the
// underlying hardware tracer for each pass, and aggregate results into XSpace.
//
// Unlike single-pass ProfilerInterface which is driven by Start()/Stop(),
// MultiPassProfilerInterface is driven by ProfilerPasses via NeedMorePasses(),
// StartPass(), and StopPass(), optionally with PushRange() / PopRange() to
// delimit profiling scopes within a pass.
class MultiPassProfilerInterface : public ProfilerInterface {
 public:
  // Returns true if there are more passes to profile.
  virtual bool NeedMorePasses() = 0;

  // Starts a new profiling pass.
  virtual absl::Status StartPass() = 0;

  // Pushes a named range to delimit profiling regions within the active pass.
  virtual absl::Status PushRange(absl::string_view name) = 0;

  // Pops the innermost named range within the active pass.
  virtual absl::Status PopRange() = 0;

  // Stops the current profiling pass.
  virtual absl::Status StopPass() = 0;

  absl::Status Start() override { return absl::OkStatus(); }

  absl::Status Stop() override { return absl::OkStatus(); }
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_
