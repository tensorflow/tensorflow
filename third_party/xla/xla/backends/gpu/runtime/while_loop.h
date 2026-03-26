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

#ifndef XLA_BACKENDS_GPU_RUNTIME_WHILE_LOOP_H_
#define XLA_BACKENDS_GPU_RUNTIME_WHILE_LOOP_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla::gpu {

// A library that supports running while loops in the XLA:GPU runtime.
//
// See StableHLO documentation for the while loop semantics:
// https://openxla.org/stablehlo/spec#while
//
// This library supports both Thunks and Commands. When a while loop executes
// thunks (via the WhileThunk), some of the nested thunks might depend on the
// loop induction variable, which is typically available in device memory.
// However, having to copy it back to host every time is very expensive.
// Instead, we keep track of the current loop iteration in a thread local
// variable, and each individual thunk (e.g., dynamic slices) can access it
// without having to synchronize with device activity. Each individual thunk is
// responsible for computing the loop induction variable from the iteration
// number.
//
// At compile time, XLA:GPU can choose to unroll the while loop, which later
// enables command buffer thunk passes to form much larger command buffers and
// launch the whole loop as one command. However, when we execute unrolled
// thunks (or record commands for them), we still need to know in which loop
// iteration a thunk or command is executing, and we rely on thread local
// storage and metadata-only thunks/commands to keep the state in thread local
// storage. Essentially we keep a thread local stack of `WhileLoopState` entries
// for all runtime loops.
//
// Example:
//
//   1. WhileThunk(cond, body); name="while.0", trip_count=2
//
// Thunk sequence after unrolling:
//
//   1. WhileEnterThunk: name="while.0", trip_count=2
//   2. SequentialThunk(cond)
//   3. SequentialThunk(body)
//   4. WhileIncrementThunk
//   5. SequentialThunk(cond)
//   6. SequentialThunk(body)
//   7. WhileExitThunk
//
// Thunks corresponding to loop Enter/Exit push/pop while loop state onto the
// thread local stack, and between every repetition of cond/body thunk
// sequences, we increment the loop iteration counter.
//
// When we record commands, we have similar metadata-only commands that update
// the thread local while loop state, so each individual command has access to
// the correct loop iteration number.

// State of a while loop execution tracked by the XLA:GPU runtime.
struct WhileLoopState {
  std::string loop_name;
  std::optional<size_t> loop_trip_count;
  size_t loop_depth = 0;
  size_t loop_iteration = 0;
};

// Returns the while loop state for the innermost loop. Returns nullptr if the
// current thread is not running within a while loop.
const WhileLoopState* IsInsideWhileLoop();

// Returns the current while loop nest as a span of WhileLoopState entries, from
// outermost to innermost loop. Returns an empty span if the current thread is
// not running within a while loop. The back of the span is the innermost loop.
//
// WARNING: The returned span points into thread-local storage and is
// invalidated by entering or exiting a while loop (i.e., constructing or
// destroying a ScopedWhileLoop). Callers must consume the span immediately and
// not hold it across such operations.
absl::Span<const WhileLoopState> IsInsideWhileLoopNest();

// An RAII helper that manages while loop state. Enters the loop upon
// construction and exits it upon destruction. This is the only public API for
// mutating while loop state.
class ScopedWhileLoop {
 public:
  explicit ScopedWhileLoop(absl::string_view loop_name,
                           std::optional<size_t> trip_count = std::nullopt);
  ~ScopedWhileLoop();

  ScopedWhileLoop(ScopedWhileLoop&&) = delete;
  ScopedWhileLoop& operator=(ScopedWhileLoop&&) = delete;

  absl::string_view loop_name() const;
  std::optional<size_t> trip_count() const;
  size_t loop_depth() const;
  size_t loop_iteration() const;

  void IncLoopIteration();

 private:
  size_t loop_depth_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_WHILE_LOOP_H_
