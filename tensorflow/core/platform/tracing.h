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

#ifndef TENSORFLOW_PLATFORM_TRACING_H_
#define TENSORFLOW_PLATFORM_TRACING_H_

// Tracing interface

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

namespace port {

class Tracing {
 public:
  // This enumeration contains the identifiers of all TensorFlow
  // threadscape events and code regions.  Threadscape assigns its
  // own identifiers at runtime when we register our events and we
  // cannot know in advance what IDs it will choose.  The "RecordEvent"
  // method and "ScopedActivity" use these event IDs for consistency
  // and remap them to threadscape IDs at runtime.  This enum is limited
  // to 64 values since we use a bitmask to configure which events are
  // enabled.  It must also be kept in step with the code in
  // "Tracing::EventCategoryString".
  enum EventCategory {
    kScheduleClosure = 0,
    kRunClosure = 1,
    kCompute = 2,
    kEventCategoryMax = 3  // sentinel - keep last
  };
  // Note: We currently only support up to 64 categories.
  static_assert(kEventCategoryMax <= 64, "only support up to 64 events");

  // Called by main programs to initialize tracing facilities
  static void Initialize();

  // Return the pathname of the directory where we are writing log files.
  static const char* LogDir();

  // Returns a non-zero identifier which can be used to correlate
  // related events.
  static inline uint64 UniqueId();

  // Returns true if a trace is in progress.  Can be used to reduce tracing
  // overheads in fast-path code.
  static inline bool IsActive();

  // Associate name with the current thread.
  static void RegisterCurrentThread(const char* name);

  // Posts an event with the supplied category and arg.
  static void RecordEvent(EventCategory category, uint64 arg);

  // Traces a region of code.  Posts a tracing "EnterCodeRegion" event
  // when created and an "ExitCodeRegion" event when destroyed.
  class ScopedActivity {
   public:
    explicit ScopedActivity(EventCategory category, uint64 arg);
    ~ScopedActivity();

   private:
#if defined(PLATFORM_GOOGLE)
    const bool enabled_;
    const int32 region_id_;
#endif

    TF_DISALLOW_COPY_AND_ASSIGN(ScopedActivity);
  };

  // Trace collection engine can be registered with this module.
  // If no engine is registered, ScopedAnnotation and TraceMe are no-ops.
  class Engine;
  static void RegisterEngine(Engine*);

  // Forward declaration of the GPU utility classes.
  class ScopedAnnotation;
  class TraceMe;

 private:
  friend class TracingTest;
  friend class ScopedAnnotation;
  friend class TraceMe;

  TF_EXPORT static std::atomic<Tracing::Engine*> tracing_engine_;
  static Tracing::Engine* engine() {
    return tracing_engine_.load(std::memory_order_acquire);
  }

  static void RegisterEvent(EventCategory id, const char* name);
  static const char* EventCategoryString(EventCategory category);

  //
  // Parses event mask expressions in 'value' of the form:
  //   expr ::= <term> (,<term>)*
  //   term ::= <event> | "!" <event>
  //   event ::= "ALL" | <wait_event> | <other_event>
  //   wait_event ::= "ENewSession" | "ECloseSession" | ...
  //   other_event ::= "Send" | "Wait" | ...
  // ALL denotes all events, <event> turns on tracing for this event, and
  // !<event> turns off tracing for this event.
  // If the expression can be parsed correctly it returns true and sets
  // the event_mask_. Otherwise it returns false and the event_mask_ is left
  // unchanged.
  static bool ParseEventMask(const char* flagname, const string& value);

  // Bit mask of enabled trace categories.
  static uint64 event_mask_;

  // Records the mappings between Threadscape IDs and the "EventCategory" enum.
  static int32 category_id_[kEventCategoryMax];
  static std::map<string, int32>* name_map_;
};

// Trace collection engine that actually implements collection.
class Tracing::Engine {
 public:
  Engine() {}
  virtual ~Engine();

  // Returns true if Tracing is currently enabled.
  virtual bool IsEnabled() const = 0;

  // Represents an active annotation.
  class Annotation {
   public:
    Annotation() {}
    virtual ~Annotation();
  };

  // Represents an active trace.
  class Tracer {
   public:
    Tracer() {}
    virtual ~Tracer();
  };

 private:
  friend class ScopedAnnotation;
  friend class TraceMe;

  // Register the specified name as an annotation on the current thread.
  // Caller should delete the result to remove the annotation.
  // Annotations from the same thread are destroyed in a LIFO manner.
  // May return nullptr if annotations are not supported.
  virtual Annotation* PushAnnotation(StringPiece name) = 0;

  // Start tracing under the specified label. Caller should delete the result
  // to stop tracing.
  // May return nullptr if tracing is not supported.
  virtual Tracer* StartTracing(StringPiece label, bool is_expensive) = 0;
  // Same as above, but implementations can avoid copying the string.
  virtual Tracer* StartTracing(string&& label, bool is_expensive) {
    return StartTracing(StringPiece(label), is_expensive);
  }

  // Backwards compatibility one arg variants (assume is_expensive=true).
  Tracer* StartTracing(StringPiece label) {
    return StartTracing(label, /*is_expensive=*/true);
  }
  Tracer* StartTracing(string&& label) {
    return StartTracing(StringPiece(label), /*is_expensive=*/true);
  }
};

// This class permits a user to apply annotation on kernels and memcpys
// when launching them. While an annotation is in scope, all activities
// within that scope get their names replaced by the annotation. The kernel
// name replacement is done when constructing the protobuf for sending out to
// a client (e.g., the stubby requestor) for both API and Activity records.
//
// Ownership: The creator of ScopedAnnotation assumes ownership of the object.
//
// Usage: {
//          ScopedAnnotation annotation("first set of kernels");
//          Kernel1<<<x,y>>>;
//          LaunchKernel2(); // Which eventually launches a cuda kernel.
//        }
// In the above scenario, the GPUProf UI would show 2 kernels with the name
// "first set of kernels" executing -- they will appear as the same kernel.
class Tracing::ScopedAnnotation {
 public:
  explicit ScopedAnnotation(StringPiece name);

  // If tracing is enabled, set up an annotation with a label of
  // "<name_part1>:<name_part2>".  Can be cheaper than the
  // single-argument constructor because the concatenation of the
  // label string is only done if tracing is enabled.
  ScopedAnnotation(StringPiece name_part1, StringPiece name_part2);

  // Returns true iff scoped annotations are active.
  static bool Enabled() {
    auto e = Tracing::engine();
    return e && e->IsEnabled();
  }

 private:
  std::unique_ptr<Engine::Annotation> annotation_;
};

// TODO(opensource): clean up the scoped classes for GPU tracing.
// This class permits user-specified (CPU) tracing activities. A trace
// activity is started when an object of this class is created and stopped
// when the object is destroyed.
class Tracing::TraceMe {
 public:
  explicit TraceMe(StringPiece name);
  TraceMe(StringPiece name, bool is_expensive);

  // If tracing is enabled, set up a traceMe with a label of
  // "<name_part1>:<name_part2>".  This can be cheaper than the
  // single-argument constructor because the concatenation of the
  // label string is only done if tracing is enabled.
  TraceMe(StringPiece name_part1, StringPiece name_part2);
  TraceMe(StringPiece name_part1, StringPiece name_part2, bool is_expensive);

 private:
  std::unique_ptr<Engine::Tracer> tracer_;
};

inline Tracing::ScopedAnnotation::ScopedAnnotation(StringPiece name) {
  auto e = Tracing::engine();
  if (e && e->IsEnabled()) {
    annotation_.reset(e->PushAnnotation(name));
  }
}

inline Tracing::ScopedAnnotation::ScopedAnnotation(StringPiece name_part1,
                                                   StringPiece name_part2) {
  auto e = Tracing::engine();
  if (e && e->IsEnabled()) {
    annotation_.reset(
        e->PushAnnotation(strings::StrCat(name_part1, ":", name_part2)));
  }
}

inline Tracing::TraceMe::TraceMe(StringPiece name) : TraceMe(name, true) {}

inline Tracing::TraceMe::TraceMe(StringPiece name, bool is_expensive) {
  auto e = Tracing::engine();
  if (e && e->IsEnabled()) {
    tracer_.reset(e->StartTracing(name, is_expensive));
  }
}

inline Tracing::TraceMe::TraceMe(StringPiece name_part1, StringPiece name_part2)
    : TraceMe(name_part1, name_part2, true) {}

inline Tracing::TraceMe::TraceMe(StringPiece name_part1, StringPiece name_part2,
                                 bool is_expensive) {
  auto e = Tracing::engine();
  if (e && e->IsEnabled()) {
    tracer_.reset(e->StartTracing(strings::StrCat(name_part1, ":", name_part2),
                                  is_expensive));
  }
}

}  // namespace port
}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/tracing_impl.h"
#else
#include "tensorflow/core/platform/default/tracing_impl.h"
#endif

#endif  // TENSORFLOW_PLATFORM_TRACING_H_
