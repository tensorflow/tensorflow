/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_INITIALIZE_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_INITIALIZE_H_

#include "xla/stream_executor/platform/platform.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_CHROMIUMOS)
#include "xla/stream_executor/platform/google/initialize.h"  // IWYU pragma: export
#else
#include "xla/stream_executor/platform/default/initialize.h"  // IWYU pragma: export
#endif

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_INITIALIZE_H_
