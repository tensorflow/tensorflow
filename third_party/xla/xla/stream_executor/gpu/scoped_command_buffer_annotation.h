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

#ifndef XLA_STREAM_EXECUTOR_GPU_SCOPED_COMMAND_BUFFER_ANNOTATION_H_
#define XLA_STREAM_EXECUTOR_GPU_SCOPED_COMMAND_BUFFER_ANNOTATION_H_

#include <type_traits>

#include "absl/strings/string_view.h"

namespace stream_executor {

// Thread-local stack of HLO annotations for the GPU command buffer nodes
// created while command buffer annotation recording is on.
//
// `CommandBufferThunk` opens the outermost scope with `Root()`, which turns
// recording on only when command buffer telemetry is enabled. Inside it,
// `CommandExecutor` pushes one scope per command with the lazy constructor, and
// stream_executor reads `GetCurrentAnnotation()` and `IsActive()` while it
// creates graph nodes. Like `tsl::profiler::ScopedAnnotation` while profiling
// is off, scopes do nothing while recording is off, so producers do no
// telemetry work unless telemetry is on.
//
// Path semantics: the retrieved `::`-joined string matches the
// `tsl::profiler::AnnotationStack` format. It keeps the outer module prefix
// (e.g. `XlaModule:#module=...#`) even for nodes within nested conditional or
// while loop bodies.
class ScopedCommandBufferAnnotation {
 public:
  // Opens the outermost scope of a command buffer recording. If `enabled`,
  // annotation recording is on for the calling thread for the lifetime of the
  // returned scope and `annotation` is its outermost frame (an empty
  // `annotation` adds no frame); otherwise the returned scope does nothing.
  // This is the only way to turn recording on. Roots nest.
  //
  // Returned by value through C++17 guaranteed copy elision: the class is
  // neither copyable nor movable, so bind the result directly, e.g.
  // `auto root = ScopedCommandBufferAnnotation::Root(...);`.
  static ScopedCommandBufferAnnotation Root(bool enabled,
                                            absl::string_view annotation);

  // Appends `annotation` (separated by `"::"`) to the calling thread's stack
  // for the lifetime of this scope while recording is on; does nothing
  // otherwise, so a producer can never turn recording on. An empty
  // `annotation` adds nothing.
  explicit ScopedCommandBufferAnnotation(absl::string_view annotation);

  // Like the above, but calls `generator` (returning anything convertible to
  // `absl::string_view`, e.g. `std::string` or `absl::string_view`) only while
  // recording is on, so producers do no work when it is off.
  template <typename Generator,
            std::enable_if_t<std::is_invocable_v<Generator&>, int> = 0>
  explicit ScopedCommandBufferAnnotation(Generator&& generator) {
    if (IsActive()) {
      // Bound to a reference so that a returned `std::string` outlives the
      // append.
      const auto& annotation = generator();
      Push(annotation);
    }
  }

  ~ScopedCommandBufferAnnotation();

  ScopedCommandBufferAnnotation(const ScopedCommandBufferAnnotation&) = delete;
  ScopedCommandBufferAnnotation& operator=(
      const ScopedCommandBufferAnnotation&) = delete;

  // Returns true while an enabled root is open on the calling thread. Readers
  // that do telemetry work without pushing (e.g. graph finalization) check
  // this.
  static bool IsActive();

  // Returns the calling thread's `"::"`-joined annotation stack, which is empty
  // while recording is off. The returned view points into the thread-local
  // buffer and is valid only until the next push or pop on this thread.
  static absl::string_view GetCurrentAnnotation();

 private:
  struct RootTag {};

  ScopedCommandBufferAnnotation(RootTag, bool enabled,
                                absl::string_view annotation);

  // Appends a non-empty `annotation` as a new frame and sets `pushed_`.
  void Push(absl::string_view annotation);

  // An enabled root: decrements the thread's root depth on destruction.
  bool is_root_ = false;
  // Pops its frame on destruction.
  bool pushed_ = false;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_SCOPED_COMMAND_BUFFER_ANNOTATION_H_
