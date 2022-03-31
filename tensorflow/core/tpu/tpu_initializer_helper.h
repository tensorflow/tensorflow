/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_TPU_INITIALIZER_HELPER_H_
#define TENSORFLOW_CORE_TPU_TPU_INITIALIZER_HELPER_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tpu {

// This will acquire a system-wide lock on behalf of the whole process. Follow
// up calls to this function will return true if the lock has been acquired and
// false if we failed to acquire the lock.
Status TryAcquireTpuLock();
// This will check the lock and then load the library.
Status FindAndLoadTpuLibrary();
// Returns arguments (e.g. flags) set in the LIBTPU_INIT_ARGS environment
// variable. The first return value is the arguments, the second return value is
// pointers to the arguments suitable for passing into the C API.
std::pair<std::vector<std::string>, std::vector<const char*>>
GetLibTpuInitArguments();

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_INITIALIZER_HELPER_H_
