/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_

#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_pimpl.h"  // IWYU pragma: export

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
