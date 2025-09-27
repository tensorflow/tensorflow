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

#include "absl/base/macros.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace ifrt {

// Future reuses `tsl::Future` as the short-term implementation.
//
// We will address the following properties in a new `Future` implementation.
//
// * Creating and destroying Future should be very cheap if no one ever awaits
// on the `Future`.
//
// * Awaiting on a `Future` should possibly be cancellable to lower overhead
// when the `Future` value woudld be no longer useful or relevant.
template <typename T = void>
using Future ABSL_DEPRECATE_AND_INLINE() = ::tsl::Future<T>;

template <typename T = void>
using Promise ABSL_DEPRECATE_AND_INLINE() = ::tsl::Promise<T>;

using ::tsl::JoinFutures;

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_FUTURE_H_
