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

#ifndef XLA_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
#define XLA_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_

#include "xla/stream_executor/platform_manager.h"

namespace stream_executor {
// The name `MultiPlatformManager` is deprecated. Please use `PlatformManager`
// instead and include `platform_manager.h`.
// TODO(hebecker): A migration is to `PlatformManager` is under way.
using MultiPlatformManager [[deprecated("Rename to PlatformManager")]] =
    PlatformManager;
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
