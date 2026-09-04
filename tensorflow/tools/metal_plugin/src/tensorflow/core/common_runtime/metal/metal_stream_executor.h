/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_STREAM_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_STREAM_EXECUTOR_H_

// Unlike the other headers in this directory, this one names no Objective-C
// types and can be included from plain C++.

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

namespace tensorflow {
namespace metal {

// Fills `se` with the Metal implementations of the StreamExecutor C API
// memory, stream, event, timer and transfer callbacks. Every function pointer
// in the struct is set; none is left optional.
void PopulateStreamExecutor(SP_StreamExecutor* se);

// Fills `timer_fns` with the Metal timer accessor. Split out because the
// platform owns the SP_TimerFns lifetime while the stream executor owns the
// SP_Timer objects themselves.
void PopulateTimerFns(SP_TimerFns* timer_fns);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_STREAM_EXECUTOR_H_
