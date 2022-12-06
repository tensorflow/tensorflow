/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_API_PROFILER_H_
#define TENSORFLOW_LITE_CORE_API_PROFILER_H_

#include <cstdint>

namespace tflite {

// A simple utility for enabling profiled event tracing in TensorFlow Lite.
class Profiler {
 public:
  // As certain Profiler instance might be only interested in certain event
  // types, we define each event type value to allow a Profiler to use
  // bitmasking bitwise operations to determine whether an event should be
  // recorded or not.
  enum class EventType {
    // Default event type, the metadata field has no special significance.
    DEFAULT = 1,

    // The event is an operator invocation and the event_metadata field is the
    // index of operator node.
    OPERATOR_INVOKE_EVENT = 1 << 1,

    // The event is an invocation for an internal operator of a TFLite delegate.
    // The event_metadata field is the index of operator node that's specific to
    // the delegate.
    DELEGATE_OPERATOR_INVOKE_EVENT = 1 << 2,

    // A delegate op invoke event that profiles a delegate op in the
    // Operator-wise Profiling section and not in the Delegate internal section.
    DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT = 1 << 3,

    // The event is a recording of runtime instrumentation such as the overall
    // TFLite runtime status, the TFLite delegate status (if a delegate
    // is applied), and the overall model inference latency etc.
    // Note, the delegate status and overall status are stored as separate
    // event_metadata fields. In particular, the delegate status is encoded
    // as DelegateStatus::full_status().
    GENERAL_RUNTIME_INSTRUMENTATION_EVENT = 1 << 4,

    // Telemetry events. Users and code instrumentations should invoke Telemetry
    // calls instead of using the following types directly.
    // See experimental/telemetry:profiler for definition of each metadata.
    //
    // A telemetry event that reports model and interpreter level events.
    TELEMETRY_EVENT = 1 << 5,
    // A telemetry event that reports model and interpreter level settings.
    TELEMETRY_REPORT_SETTINGS = 1 << 6,
    // A telemetry event that reports delegate level events.
    TELEMETRY_DELEGATE_EVENT = 1 << 7,
    // A telemetry event that reports delegate settings.
    TELEMETRY_DELEGATE_REPORT_SETTINGS = 1 << 8,
  };

  virtual ~Profiler() {}

  // Signals the beginning of an event and returns a handle to the profile
  // event. The `event_metadata1` and `event_metadata2` have different
  // interpretations based on the actual Profiler instance and the `event_type`.
  // For example, as for the 'SubgraphAwareProfiler' defined in
  // lite/core/subgraph.h, when the event_type is OPERATOR_INVOKE_EVENT,
  // `event_metadata1` represents the index of a TFLite node, and
  // `event_metadata2` represents the index of the subgraph that this event
  // comes from.
  virtual uint32_t BeginEvent(const char* tag, EventType event_type,
                              int64_t event_metadata1,
                              int64_t event_metadata2) = 0;
  // Similar w/ the above, but `event_metadata2` defaults to 0.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata) {
    return BeginEvent(tag, event_type, event_metadata, /*event_metadata2*/ 0);
  }

  // Signals an end to the specified profile event with 'event_metadata's, This
  // is useful when 'event_metadata's are not available when the event begins
  // or when one wants to overwrite the 'event_metadata's set at the beginning.
  virtual void EndEvent(uint32_t event_handle, int64_t event_metadata1,
                        int64_t event_metadata2) {
    // By default discards the metadata.
    EndEvent(event_handle);
  }
  // Signals an end to the specified profile event.
  virtual void EndEvent(uint32_t event_handle) = 0;

  // Appends an event of type 'event_type' with 'tag' and 'event_metadata'
  // which ran for elapsed_time.
  // Note:
  // In cases were ProfileSummarizer and tensorflow::StatsCalculator are used
  // they assume the value is in "usec", if in any case subclasses
  // didn't put usec, then the values are not meaningful.
  // TODO(karimnosseir): karimnosseir: Revisit and make the function more clear.
  void AddEvent(const char* tag, EventType event_type, uint64_t elapsed_time,
                int64_t event_metadata) {
    AddEvent(tag, event_type, elapsed_time, event_metadata,
             /*event_metadata2*/ 0);
  }

  // Adds a profiler event.
  // `metric` field has different intreptation based on `event_type`.
  // e.g. it means elapsed time for [DELEGATE_]OPERATOR_INVOKE_EVENT types,
  // and interprets as source and status code for TELEMETRY_[DELEGATE_]EVENT
  // event types. If the concrete profiler does not provide an implementation,
  // does nothing.
  // TODO(b/241982974): Clean up dependencies and make it pure virtual.
  virtual void AddEvent(const char* tag, EventType event_type, uint64_t metric,
                        int64_t event_metadata1, int64_t event_metadata2) {}

  // Adds a profiler event with data.
  // Data will be a const TelemetrySettings* for TELEMETRY_REPORT_SETTINGS
  // and TELEMETRY_DELEGATE_REPORT_SETTINGS.
  // If the concrete profiler does not provide an implementation, does nothing.
  // TODO(b/241982974): Clean up dependencies and make it pure virtual.
  virtual void AddEventWithData(const char* tag, EventType event_type,
                                const void* data) {}

 protected:
  friend class ScopedProfile;
};

// Adds a profile event to `profiler` that begins with the construction
// of the object and ends when the object goes out of scope.
// The lifetime of tag should be at least the lifetime of `profiler`.
// `profiler` may be null, in which case nothing is profiled.
class ScopedProfile {
 public:
  ScopedProfile(Profiler* profiler, const char* tag,
                Profiler::EventType event_type = Profiler::EventType::DEFAULT,
                int64_t event_metadata = 0)
      : profiler_(profiler), event_handle_(0) {
    if (profiler) {
      event_handle_ = profiler_->BeginEvent(tag, event_type, event_metadata);
    }
  }

  ~ScopedProfile() {
    if (profiler_) {
      profiler_->EndEvent(event_handle_);
    }
  }

 protected:
  Profiler* profiler_;
  uint32_t event_handle_;
};

class ScopedOperatorProfile : public ScopedProfile {
 public:
  ScopedOperatorProfile(Profiler* profiler, const char* tag, int node_index)
      : ScopedProfile(profiler, tag, Profiler::EventType::OPERATOR_INVOKE_EVENT,
                      static_cast<uint32_t>(node_index)) {}
};

class ScopedDelegateOperatorProfile : public ScopedProfile {
 public:
  ScopedDelegateOperatorProfile(Profiler* profiler, const char* tag,
                                int node_index)
      : ScopedProfile(profiler, tag,
                      Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT,
                      static_cast<uint32_t>(node_index)) {}
};

// Similar to ScopedProfile but has extra event metadata for EndEvent.
class ScopedRuntimeInstrumentationProfile {
 public:
  ScopedRuntimeInstrumentationProfile(Profiler* profiler, const char* tag)
      : profiler_(profiler), event_handle_(0) {
    if (profiler) {
      event_handle_ = profiler_->BeginEvent(
          tag, Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT,
          /*event_metadata=*/-1);
    }
  }

  void set_runtime_status(int64_t delegate_status, int64_t interpreter_status) {
    if (profiler_) {
      delegate_status_ = delegate_status;
      interpreter_status_ = interpreter_status;
    }
  }

  ~ScopedRuntimeInstrumentationProfile() {
    if (profiler_) {
      profiler_->EndEvent(event_handle_, delegate_status_, interpreter_status_);
    }
  }

 private:
  Profiler* profiler_ = nullptr;
  uint32_t event_handle_ = 0;
  int64_t delegate_status_ = 0;
  int64_t interpreter_status_ = 0;
};

}  // namespace tflite

#define TFLITE_VARNAME_UNIQ_IMPL(name, ctr) name##ctr
#define TFLITE_VARNAME_UNIQ(name, ctr) TFLITE_VARNAME_UNIQ_IMPL(name, ctr)

#define TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler, tag)          \
  tflite::ScopedProfile TFLITE_VARNAME_UNIQ(_profile_, __COUNTER__)( \
      (profiler), (tag))

#define TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE(profiler, tag, node_index)     \
  tflite::ScopedOperatorProfile TFLITE_VARNAME_UNIQ(_profile_, __COUNTER__)( \
      (profiler), (tag), (node_index))

#define TFLITE_SCOPED_DELEGATE_OPERATOR_PROFILE(profiler, tag, node_index) \
  tflite::ScopedDelegateOperatorProfile TFLITE_VARNAME_UNIQ(               \
      _profile_, __COUNTER__)((profiler), (tag), (node_index))

#define TFLITE_ADD_RUNTIME_INSTRUMENTATION_EVENT(                          \
    profiler, tag, event_metadata1, event_metadata2)                       \
  do {                                                                     \
    if (profiler) {                                                        \
      const auto handle = profiler->BeginEvent(                            \
          tag, Profiler::EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT, \
          event_metadata1, event_metadata2);                               \
      profiler->EndEvent(handle);                                          \
    }                                                                      \
  } while (false);

#endif  // TENSORFLOW_LITE_CORE_API_PROFILER_H_
