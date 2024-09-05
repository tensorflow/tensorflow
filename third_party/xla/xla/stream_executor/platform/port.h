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

// IWYU pragma: private, include "xla/stream_executor/stream_executor.h"

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_PORT_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_PORT_H_

#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

namespace stream_executor {

using tsl::int16;
using tsl::int32;
using tsl::int8;

using tsl::uint16;
using tsl::uint32;
using tsl::uint64;
using tsl::uint8;

#if !defined(PLATFORM_GOOGLE)
using std::string;
#endif

#define SE_FALLTHROUGH_INTENDED TF_FALLTHROUGH_INTENDED

}  // namespace stream_executor

// DEPRECATED: directly use the macro implementation instead.
#define SE_DISALLOW_COPY_AND_ASSIGN TF_DISALLOW_COPY_AND_ASSIGN

#define SE_MUST_USE_RESULT TF_MUST_USE_RESULT
#define SE_PREDICT_TRUE TF_PREDICT_TRUE
#define SE_PREDICT_FALSE TF_PREDICT_FALSE

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_PORT_H_
