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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_LIB_ARRAY_SLICE_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_LIB_ARRAY_SLICE_H_

#include "absl/types/span.h"

namespace stream_executor {
namespace port {

template <typename T>
using ArraySlice = absl::Span<const T>;  // non-absl ok
template <typename T>
using MutableArraySlice = absl::Span<T>;

}  // namespace port
}  // namespace stream_executor

namespace perftools {
namespace gputools {

// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(b/77980417): Remove this once we've
// completed the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_LIB_ARRAY_SLICE_H_
