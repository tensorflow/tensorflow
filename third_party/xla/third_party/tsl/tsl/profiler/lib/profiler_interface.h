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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_

#include <cstddef>

#include "absl/status/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

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

  // Pulls collected profile data into arbitrary raw memory.
  // Size refers to the amount of data memory `ptr` points to.
  virtual absl::Status Consume(void* ptr, size_t size) {
    return absl::UnimplementedError("Consume not implemented");
  }

  // Serializes collected profile data into a wire-transferrable format.
  virtual absl::Status Serialize(void* ptr, size_t size, void* output,
                                 size_t output_size) {
    return absl::UnimplementedError("Serialize not implemented");
  }
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_
