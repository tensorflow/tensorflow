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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/xfeed_manager.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "tsl/platform/statusor.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

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
    kCall,
    kCopy,
    kConditional,
    kDot,
    kInfeed,
    kKernel,
    kOutfeed,
    kRngGetAndUpdateState,
    kWhile,
  };

  struct Info {
    std::string op_name;
    std::string module_name;
    int64_t module_id;
  };

  virtual ~Thunk() = default;

  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  explicit Thunk(Kind kind, Info info) : kind_(kind), info_(std::move(info)) {}

  Kind kind() const { return kind_; }
  const Info& info() const { return info_; }

  static std::string_view KindToString(Kind kind);

  // Returns the list of buffers used by a thunk. Thunk executor relies on this
  // information to execute thunks concurrently and to avoid data races.
  using BufferUses = absl::InlinedVector<BufferUse, 4>;
  virtual BufferUses buffer_uses() const = 0;

  //===--------------------------------------------------------------------===//
  // HostKernels
  //===--------------------------------------------------------------------===//

  // Interface for finding host kernels (function pointers with host kernel API)
  // by name. At run time this is typically backed by an LLVM jit compiler that
  // compiles LLVM IR to executables on demand.
  class HostKernels {
   public:
    virtual ~HostKernels() = default;

    virtual absl::StatusOr<SE_HOST_Kernel*> Find(std::string_view name) = 0;
  };

  //===--------------------------------------------------------------------===//
  // ExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Execute. Execute is responsible for launching "work"
  // on device, i.e., it launches host kernels, calls into libraries, etc.
  struct ExecuteParams {
    HostKernels* host_kernels = nullptr;
    const BufferAllocations* buffer_allocations = nullptr;
    runtime::XfeedManager* xfeed = nullptr;
    const Eigen::ThreadPoolDevice* intra_op_threadpool = nullptr;
  };

  // An execute event that becomes ready when all tasks are completed.
  using ExecuteEvent = tsl::Chain;

  // Returns non-reference-counted async value ref for thunks executed in the
  // caller thread to avoid reference counting overhead.
  static tsl::AsyncValueRef<ExecuteEvent> OkExecuteEvent();

  // Thunk execution must be asynchronous and never block the caller thread,
  // especially waiting for work submitted into the `intra_op_threadpool`,
  // because thunks themselves are executed on the same thread pool.
  //
  // Thunk execution completion must be reported via the `ExecuteEvent`.
  virtual tsl::AsyncValueRef<ExecuteEvent> Execute(
      const ExecuteParams& params) = 0;

 protected:
  // Encodes thunk info into the TraceMe compatible format.
  std::string TraceMeEncode() const;

 private:
  Kind kind_;
  Info info_;
};

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// A sequence of thunks to execute.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  ThunkSequence() = default;

  // Returns an empty thunk sequence.
  static ThunkSequence Empty() { return ThunkSequence(); }

  // Returns a thunk sequence that contains a single thunk of type `T`. Uses
  // factory constructor `T::Create()` to create the thunk.
  template <typename T, typename... Args>
  static absl::StatusOr<ThunkSequence> Of(Args&&... args) {
    static_assert(std::is_base_of_v<Thunk, T>,
                  "ThunkSequence::Of() requires `T` to be a `Thunk` subclass.");
    TF_ASSIGN_OR_RETURN(auto thunk, T::Create(std::forward<Args>(args)...));
    return ThunkSequence(std::move(thunk));
  }

  using BufferUses = Thunk::BufferUses;
  BufferUses buffer_uses() const;

  void Append(ThunkSequence other);

 private:
  explicit ThunkSequence(std::unique_ptr<Thunk> thunk);
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_H_
