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

// The StreamExecutor is a single-device abstraction for:
//
// * Loading/launching data-parallel-kernels
// * Invoking pre-canned high-performance library routines (like matrix
//   multiply)

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_

#include "tensorflow/compiler/xla/stream_executor/device_description.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/device_options.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/event.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/kernel.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/kernel_spec.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/launch_dim.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/platform.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/stream.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"  // IWYU pragma: export
#include "tensorflow/compiler/xla/stream_executor/timer.h"  // IWYU pragma: export

namespace perftools {
namespace gputools {

// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(b/77980417): Remove this once we've
// completed the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
