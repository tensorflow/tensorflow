/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_THUNK_H_

#include <memory>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"

namespace xla::cpu {

// WARNING: This is under construction. Long term plan for XLA is to unify
// runtimes between different backends and have a shared Thunk interface,
// however for now we chose to have separate Thunk implementations in xla::cpu
// and xla::gpu namespaces with a plan to unify them in the future.

// Thunk is the basic unit of execution for the XLA CPU runtime.
//
// This is thread-compatible. Thunk implementation should expect that it will be
// called concurrently from multiple threads, for different run ids and for
// different devices. For partitioned XLA programs the expectation is that all
// local participants execute simultaneously on different threads and coordinate
// resource acquisition via rendezvous.
//
// This is XLA CPU's counterpart of the XLA GPU runtime Thunk.
class Thunk {
 public:
  enum class Kind {
    kCopy,
  };

  virtual ~Thunk() = default;

  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  explicit Thunk(Kind kind) : kind_(kind) {}

  Kind kind() const { return kind_; }

  static std::string_view KindToString(Kind kind);

  //===--------------------------------------------------------------------===//
  // ExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Execute. Execute is responsible for launching "work"
  // on device, i.e., it launches host kernels, calls into libraries, etc.
  struct ExecuteParams {
    const BufferAllocations* buffer_allocations = nullptr;
  };

  virtual absl::Status Execute(const ExecuteParams& params) = 0;

 private:
  Kind kind_;
};

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// A sequence of thunks to execute.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  ThunkSequence() = default;
  explicit ThunkSequence(std::unique_ptr<Thunk> thunk);

  // Return a ThunkSequence that contains a single thunk of type `T`.
  template <typename T, typename... Args>
  static ThunkSequence Of(Args&&... args) {
    static_assert(std::is_base_of_v<Thunk, T>,
                  "ThunkSequence::Of() requires `T` to be a `Thunk` subclass.");
    return ThunkSequence(std::make_unique<T>(std::forward<Args>(args)...));
  }

  static ThunkSequence Empty() { return ThunkSequence(); }

  absl::Status Execute(const Thunk::ExecuteParams& params);

  void Append(ThunkSequence other);
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_H_
