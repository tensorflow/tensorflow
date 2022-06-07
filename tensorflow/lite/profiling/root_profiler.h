/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PROFILING_ROOT_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_ROOT_PROFILER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"

namespace tflite {
namespace profiling {

/// A root profiler instance installed in TFLite runtime.
/// It's capable to dispatching profiling events to all child profilers attached
/// to it. Child profilers can either accept for discard the events based on the
/// event type.
class RootProfiler : public Profiler {
 public:
  RootProfiler() = default;
  ~RootProfiler() override = default;

  // Not copiable.
  RootProfiler(const RootProfiler&) = delete;
  RootProfiler& operator=(const RootProfiler&) = delete;

  // Movable.
  RootProfiler(RootProfiler&&) = default;
  RootProfiler& operator=(RootProfiler&&) = default;

  /// Adds a profiler to root profiler.
  /// Added `profiler` should not be nullptr or it will be ignored.
  /// Caller must retains the ownership. The lifetime should exceed the
  /// lifetime of the RootProfiler.
  void AddProfiler(Profiler* profiler);

  /// Adds a profiler to RootProfiler.
  /// Added `profiler` should not be nullptr or it will be ignored.
  /// Transfers the ownership of `profiler` to RootProfiler.
  void AddProfiler(std::unique_ptr<Profiler>&& profiler);

  /// Signals the beginning of an event to all child profilers.
  /// The `tag`, `event_metadata1` and `event_metadata2` arguments have
  /// different interpretations based on the actual Profiler instance
  /// and the `event_type`.
  /// Returns a handle to the profile event which can be used in a later
  /// `EndEvent` call.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  /// Signals an end to the specified profile event to all child profilers with
  /// 'event_metadata's.
  /// An invalid event handle (e.g. not a value returned from BeginEvent call or
  /// a handle invalidated by RemoveChildProfilers) will be ignored.
  void EndEvent(uint32_t event_handle, int64_t event_metadata1,
                int64_t event_metadata2) override;
  /// Signals an end to the specified profile event to all child profilers.
  /// An invalid event handle (e.g. not a value returned from BeginEvent call or
  /// a handle invalidated by RemoveChildProfilers) will be ignored.
  void EndEvent(uint32_t event_handle) override;

  /// Appends an event of type 'event_type' with 'tag' and 'event_metadata'
  /// which ran for elapsed_time.
  /// The `tag`, `event_metadata1` and `event_metadata2` arguments have
  /// different interpretations based on the actual Profiler instance
  /// and the `event_type`.
  void AddEvent(const char* tag, EventType event_type, uint64_t elapsed_time,
                int64_t event_metadata1, int64_t event_metadata2) override;

  /// Removes all child profilers and releases the child profiler if it's owned
  /// by the root profiler. Also invalidates all event handles generated
  /// from previous `BeginEvent` calls.
  void RemoveChildProfilers();

 private:
  uint32_t next_event_id_ = 1;
  std::vector<std::unique_ptr<Profiler>> owned_profilers_;
  std::vector<Profiler*> profilers_;
  std::map<uint32_t, std::vector<uint32_t>> events_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_ROOT_PROFILER_H_
