/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_ID_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_ID_H_

#include "xla/stream_executor/platform.h"

namespace tensorflow {
namespace tpu {

::stream_executor::Platform::Id GetTpuPlatformId();

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_PLATFORM_ID_H_
