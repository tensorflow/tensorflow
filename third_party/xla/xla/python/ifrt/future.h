/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_FUTURE_H_
#define XLA_PYTHON_IFRT_FUTURE_H_

#include "xla/pjrt/pjrt_client.h"
#include "xla/status.h"

namespace xla {
namespace ifrt {

// Future reuses `xla::PjRtFuture` as the short-term implementation.
//
// We will address the following properties in a new `Future` implementation.
//
// * Creating and destroying Future should be very cheap if no one ever awaits
// on the `Future`.
//
// * Awaiting on a `Future` should possibly be cancellable to lower overhead
// when the `Future` value woudld be no longer useful or relevant.
//
// * Ideally, there should be a move-only version of `Future`, which will enable
// (1) no reference counting of `Future`s sharing the same `Promise` and (2)
// safe mutable access to the value when the `Future` becomes ready, including
// moving the value out of the `Future`/`Promise`.
template <typename T>
using Future = ::xla::PjRtFuture<T>;

template <typename T>
using Promise = typename ::xla::PjRtFuture<T>::Promise;

// TODO(b/333538339): Add JoinFutures API to PjRtFuture.

// Returns a `Future` that aggregates the return status of all `Future`s.
Future<Status> JoinFutures(absl::Span<Future<Status>> futures);

// TODO(b/333538339): Use Future<> with implicit Status in IFRT APIs. For now
// this is a workaround to convert between different error semantics.S
Future<Status> JoinFutures(absl::Span<Future<void>> futures);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_FUTURE_H_
