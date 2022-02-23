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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_FUTURE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_FUTURE_H_

#include <functional>
#include <utility>

#include "absl/types/span.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace xla {

// Helpers for using PjRtFutures.
struct PjRtFutureHelpers {
 public:
  // Keys that are returned by an implementation-specific handler when a client
  // starts to block on a promise.
  //
  // For now, contains a single UID that can be used to identify a TraceMe, but
  // made extensible to allow support for other profilers such as endoscope.
  struct ProfilingKeys {
    uint64_t traceme_context_id = -1;
  };
  // Signature of handler called by the PjRtFuture class before it starts to
  // block a thread.
  using OnBlockStartFn = std::function<ProfilingKeys()>;
  // Signature of handler called by the PjRtFuture class after it finishes
  // blocking a thread.
  using OnBlockEndFn = std::function<void(ProfilingKeys)>;
};

// PjRtFuture<T> is a simple future that is returned by PjRt APIs that
// enqueue asynchronous work, reporting a value of type T (frequently T=Status)
// when the work is complete.
//
// PjRtFuture can be used by the client to wait for work to complete, either via
// a blocking call or a callback.
//
// The implementation wraps a TFRT AsyncValueRef<T>, but we prefer to
// encapsulate the AVR rather than returning it directly for two reasons.
//
// First, we want to retain portability in case a future implementation moves
// away from AsyncValueRef ---- we don't want clients to call arbitrary
// AsyncValueRef APIs.
//
// Second, we want to export different semantics, for
// example we block without the client supplying a HostContext, and support
// integration between blocking and profiling (e.g., TraceMe).
//
// There are two ways to construct a PjRtFuture, one used by clients that
// natively use TFRT, which already have a HostContext and import APIs for
// constructing AsyncValueRefs; and another that avoids exposing TFRT APIs and
// can be used by non-TFRT clients.
template <class T>
class PjRtFuture {
 public:
  // Wrapper for AsyncValueRef<T> that can be used by clients that don't
  // natively use TFRT.
  struct Promise {
   public:
    // Creates an empty promise with !this == true.
    explicit Promise() = default;
    Promise(Promise&& other) = default;
    Promise(const Promise& other) : avr(other.avr.CopyRef()) {}
    Promise& operator=(const Promise& other) {
      avr = other.avr.CopyRef();
      return *this;
    }
    bool operator!() { return !avr; }

    // Sets the value of the promise. Must be called at most once.
    //
    // After Set is called, value will be delivered to waiters on the parent
    // PjRtFuture, via blocking or callbacks.
    void Set(T value) { avr.emplace(std::move(value)); }

   private:
    friend class PjRtFuture<T>;
    explicit Promise(tfrt::AsyncValueRef<T> ref) : avr(std::move(ref)) {}
    // The underlying TFRT value that can be waited on.
    tfrt::AsyncValueRef<T> avr;
  };

  // Returns a Promise that can be used to construct a PjRtFuture, and then Set
  // later.
  //
  // Used by clients that do not use TFRT natively.
  static Promise CreatePromise() {
    return Promise(tfrt::MakeUnconstructedAsyncValueRef<T>());
  }

  // Constructor for an already-available PjRtFuture.
  //
  // Typically used to eagerly return error values when async work will not
  // be enqueued, e.g., due to invalid arguments.
  explicit PjRtFuture(T t)
      : promise_(tfrt::MakeAvailableAsyncValueRef<T>(t)),
        on_block_start_([]() { return PjRtFutureHelpers::ProfilingKeys(); }),
        on_block_end_([](PjRtFutureHelpers::ProfilingKeys) {}),
        host_ctx_(nullptr) {}

  // Constructor used by clients that natively use TFRT and already have a
  // host_ctx that should be used for awaiting promises.
  //
  // on_block_start is called before Await starts to block.
  // on_block_end is called after Await finishes blocking.
  explicit PjRtFuture(
      tfrt::HostContext* host_ctx, tfrt::AsyncValueRef<T> promise,
      PjRtFutureHelpers::OnBlockStartFn on_block_start =
          []() { return PjRtFutureHelpers::ProfilingKeys(); },
      PjRtFutureHelpers::OnBlockEndFn on_block_end =
          [](PjRtFutureHelpers::ProfilingKeys) {})
      : promise_(std::move(promise)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)),
        host_ctx_(host_ctx) {}

  // Constructor used by clients that don't natively use TFRT and want to use
  // the wrapped PjRtFuture<T>::Promise class and block without using
  // HostContext.
  //
  // on_block_start is called before Await starts to block.
  // on_block_end is called after Await finishes blocking.
  explicit PjRtFuture(
      Promise promise,
      PjRtFutureHelpers::OnBlockStartFn on_block_start =
          []() { return PjRtFutureHelpers::ProfilingKeys(); },
      PjRtFutureHelpers::OnBlockEndFn on_block_end =
          [](PjRtFutureHelpers::ProfilingKeys) {})
      : promise_(std::move(promise.avr)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)),
        host_ctx_(nullptr) {}

  // Blocks the calling thread until the promise is ready, then returns the
  // final value.
  T Await() {
    if (!promise_.IsAvailable()) {
      const auto keys = on_block_start_();
      if (host_ctx_) {
        host_ctx_->Await({promise_.CopyRCRef()});
      } else {
        tfrt::Await({promise_.GetAsyncValue()});
      }
      on_block_end_(keys);
    }
    DCHECK(promise_.IsConcrete());
    return *promise_;
  }

  // Registers callback to be called once the promise is ready, with the final
  // value.
  //
  // callback may be called on an internal system thread or the calling thread.
  // The client should avoid any potentially re-entrant API calls within the
  // callback, for example by using the callback to enqueue work on a
  // client-owned threadpool.
  void OnReady(std::function<void(T)> callback) {
    promise_.AndThen(
        [promise = promise_.CopyRef(), callback = std::move(callback)]() {
          DCHECK(promise.IsConcrete());
          callback(*promise);
        });
  }

 private:
  // Wrapped object to wait on.
  tfrt::AsyncValueRef<T> promise_;
  // Function that is called before a thread starts blocking on the promise.
  PjRtFutureHelpers::OnBlockStartFn on_block_start_;
  // Function that is called after a thread finishes blocking on the promise.
  PjRtFutureHelpers::OnBlockEndFn on_block_end_;
  // Used only to await promise_.
  tfrt::HostContext* host_ctx_;  // not owned
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_FUTURE_H_
