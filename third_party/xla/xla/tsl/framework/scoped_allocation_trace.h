/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TSL_FRAMEWORK_SCOPED_ALLOCATION_TRACE_H_
#define XLA_TSL_FRAMEWORK_SCOPED_ALLOCATION_TRACE_H_

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tsl {

// Thread-local metadata for passing higher-level allocation details down to
// allocators. XLA/PJRT code can describe what an allocation represents, and
// allocators may snapshot that stack to connect low-level chunks back to
// high-level execution state during OOM diagnostics.
//
// This complements ScopedMemoryDebugAnnotation: that API exposes current
// pending op/shape metadata for memory profiling, while this API keeps explicit
// allocation trace frames for allocator diagnostics. This is scoped to the
// current thread and does not automatically propagate to other threads. This
// differs from third_party/tsl/tsl/platform/context.h, whose context can be
// automatically captured and propagated through XLA thread pools.
//
// Allocator implementations may optionally snapshot Current() when a buffer
// becomes live and attach it to internal metadata. Callers should not assume
// every allocator records it.
//
// Example:
//
//   ScopedAllocationTrace exec_scope(
//       "xla.execute", {{"executable", executable_name}});
//
//   void* ptr = allocator->AllocateRaw(alignment, bytes);
//
class ScopedAllocationTrace {
 public:
  // Key/value pair encoded into an allocation trace frame.
  struct Arg {
    Arg(absl::string_view key,
        const absl::AlphaNum& value ABSL_ATTRIBUTE_LIFETIME_BOUND);

    Arg(const Arg&) = delete;
    void operator=(const Arg&) = delete;

    absl::string_view key;
    absl::string_view value;
  };

  // Single allocation trace scope frame.
  struct Frame {
    explicit Frame(absl::string_view name);

    Frame& Add(absl::string_view key,
               const absl::AlphaNum& value ABSL_ATTRIBUTE_LIFETIME_BOUND);

    std::string name;
    std::vector<std::pair<std::string, std::string>> args;
  };

  // Copy of the current thread-local trace frame stack.
  struct Snapshot {
    explicit Snapshot(std::vector<Frame> frames);

    std::vector<Frame> frames;
  };

  explicit ScopedAllocationTrace(absl::string_view name,
                                 std::initializer_list<Arg> args = {});
  explicit ScopedAllocationTrace(Frame frame);

  ScopedAllocationTrace(ScopedAllocationTrace&&) = delete;

  ~ScopedAllocationTrace();

  // Returns a copy of the current thread's annotation stack. The returned
  // snapshot is independent from later scope changes and can have no frames.
  static Snapshot Current();
};

}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_SCOPED_ALLOCATION_TRACE_H_
