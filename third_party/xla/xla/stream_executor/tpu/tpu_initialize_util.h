/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_INITIALIZE_UTIL_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_INITIALIZE_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace tensorflow {
namespace tpu {

// This will acquire a system-wide lock on behalf of the whole process. Follow
// up calls to this function will return true if the lock has been acquired and
// false if we failed to acquire the lock.
absl::Status TryAcquireTpuLock();  // TENSORFLOW_STATUS_OK

// Returns arguments (e.g. flags) set in the LIBTPU_INIT_ARGS environment
// variable. The first return value is the arguments, the second return value is
// pointers to the arguments suitable for passing into the C API.
std::pair<std::vector<std::string>, std::vector<const char*>>
GetLibTpuInitArguments();

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_INITIALIZE_UTIL_H_
