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
#include <atomic>
#include <map>
#include <memory>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tracing {

// This enumeration contains the identifiers of all TensorFlow CPU profiler
// events. It must be kept in sync with the code in GetEventCategoryName().
enum struct EventCategory : unsigned {
  kScheduleClosure = 0,
  kRunClosure = 1,
  kCompute = 2,
  kNumCategories = 3  // sentinel - keep last
};
constexpr unsigned GetNumEventCategories() {
  return static_cast<unsigned>(EventCategory::kNumCategories);
}
const char* GetEventCategoryName(EventCategory);

// Interface for CPU profiler events.
class EventCollector {
 public:
  virtual ~EventCollector() {}
  virtual void RecordEvent(uint64 arg) const = 0;
  virtual void StartRegion(uint64 arg) const = 0;
  virtual void StopRegion() const = 0;

  // Annotates the current thread with a name.
  static void SetCurrentThreadName(const char* name);
  // Returns whether event collection is enabled.
  static bool IsEnabled();

 private:
  friend void SetEventCollector(EventCategory, const EventCollector*);
  friend const EventCollector* GetEventCollector(EventCategory);

  static std::array<const EventCollector*, GetNumEventCategories()> instances_;
};
// Set the callback for RecordEvent and ScopedRegion of category.
// Not thread safe. Only call while EventCollector::IsEnabled returns false.
void SetEventCollector(EventCategory category, const EventCollector* collector);

// Returns the callback for RecordEvent and ScopedRegion of category if
// EventCollector::IsEnabled(), otherwise returns null.
inline const EventCollector* GetEventCollector(EventCategory category) {
  if (EventCollector::IsEnabled()) {
    return EventCollector::instances_[static_cast<unsigned>(category)];
  }
  return nullptr;
}

// Returns a unique id to pass to RecordEvent/ScopedRegion. Never returns zero.
uint64 GetUniqueArg();

// Returns an id for name to pass to RecordEvent/ScopedRegion.
uint64 GetArgForName(StringPiece name);

// Records an atomic event through the currently registered EventCollector.
inline void RecordEvent(EventCategory category, uint64 arg) {
  if (auto collector = GetEventCollector(category)) {
    collector->RecordEvent(arg);
  }
}

// Records an event for the duration of the instance lifetime through the
// currently registered EventCollector.
class ScopedRegion {
  ScopedRegion(ScopedRegion&) = delete;             // Not copy-constructible.
  ScopedRegion& operator=(ScopedRegion&) = delete;  // Not assignable.

 public:
  ScopedRegion(ScopedRegion&& other) noexcept  // Move-constructible.
      : collector_(other.collector_) {
    other.collector_ = nullptr;
  }

  ScopedRegion(EventCategory category, uint64 arg)
      : collector_(GetEventCollector(category)) {
    if (collector_) {
      collector_->StartRegion(arg);
    }
  }

  // Same as ScopedRegion(category, GetUniqueArg()), but faster if
  // EventCollector::IsEnaled() returns false.
  ScopedRegion(EventCategory category)
      : collector_(GetEventCollector(category)) {
    if (collector_) {
      collector_->StartRegion(GetUniqueArg());
    }
  }

  // Same as ScopedRegion(category, GetArgForName(name)), but faster if
  // EventCollector::IsEnaled() returns false.
  ScopedRegion(EventCategory category, StringPiece name)
      : collector_(GetEventCollector(category)) {
    if (collector_) {
      collector_->StartRegion(GetArgForName(name));
    }
  }

  ~ScopedRegion() {
    if (collector_) {
      collector_->StopRegion();
    }
  }

  bool IsEnabled() const { return collector_ != nullptr; }

 private:
  const EventCollector* collector_;
};

// Interface for accelerator profiler annotations.
class TraceCollector {
 public:
  class Handle {
   public:
    virtual ~Handle() {}
  };

  virtual ~TraceCollector() {}
  virtual std::unique_ptr<Handle> CreateAnnotationHandle(
      StringPiece name_part1, StringPiece name_part2) const = 0;
  virtual std::unique_ptr<Handle> CreateActivityHandle(
      StringPiece name_part1, StringPiece name_part2,
      bool is_expensive) const = 0;

  // Returns true if this annotation tracing is enabled for any op.
  virtual bool IsEnabledForAnnotations() const = 0;

  // Returns true if this activity handle tracking is enabled for an op of the
  // given expensiveness.
  virtual bool IsEnabledForActivities(bool is_expensive) const = 0;

 protected:
  static string ConcatenateNames(StringPiece first, StringPiece second);

 private:
  friend void SetTraceCollector(const TraceCollector*);
  friend const TraceCollector* GetTraceCollector();
};
// Set the callback for ScopedAnnotation and ScopedActivity.
void SetTraceCollector(const TraceCollector* collector);
// Returns the callback for ScopedAnnotation and ScopedActivity.
const TraceCollector* GetTraceCollector();

// Adds an annotation to all activities for the duration of the instance
// lifetime through the currently registered TraceCollector.
//
// Usage: {
//          ScopedAnnotation annotation("my kernels");
//          Kernel1<<<x,y>>>;
//          LaunchKernel2(); // Launches a CUDA kernel.
//        }
// This will add 'my kernels' to both kernels in the profiler UI
class ScopedAnnotation {
 public:
  explicit ScopedAnnotation(StringPiece name)
      : ScopedAnnotation(name, StringPiece()) {}

  // If tracing is enabled, add a name scope of
  // "<name_part1>:<name_part2>".  This can be cheaper than the
  // single-argument constructor because the concatenation of the
  // label string is only done if tracing is enabled.
  ScopedAnnotation(StringPiece name_part1, StringPiece name_part2)
      : handle_([&] {
          auto trace_collector = GetTraceCollector();
          return trace_collector ? trace_collector->CreateAnnotationHandle(
                                       name_part1, name_part2)
                                 : nullptr;
        }()) {}

  bool IsEnabled() const { return static_cast<bool>(handle_); }

 private:
  std::unique_ptr<TraceCollector::Handle> handle_;
};

// Adds an activity through the currently registered TraceCollector.
// The activity starts when an object of this class is created and stops when
// the object is destroyed.
class ScopedActivity {
 public:
  explicit ScopedActivity(StringPiece name, bool is_expensive = true)
      : ScopedActivity(name, StringPiece(), is_expensive) {}

  // If tracing is enabled, set up an activity with a label of
  // "<name_part1>:<name_part2>".  This can be cheaper than the
  // single-argument constructor because the concatenation of the
  // label string is only done if tracing is enabled.
  ScopedActivity(StringPiece name_part1, StringPiece name_part2,
                 bool is_expensive = true)
      : handle_([&] {
          auto trace_collector = GetTraceCollector();
          return trace_collector ? trace_collector->CreateActivityHandle(
                                       name_part1, name_part2, is_expensive)
                                 : nullptr;
        }()) {}

  bool IsEnabled() const { return static_cast<bool>(handle_); }

 private:
  std::unique_ptr<TraceCollector::Handle> handle_;
};

// Return the pathname of the directory where we are writing log files.
const char* GetLogDir();

}  // namespace tracing
}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/tracing_impl.h"
#else
#include "tensorflow/core/platform/default/tracing_impl.h"
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_TRACING_H_
