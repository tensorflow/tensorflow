/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/rocm_rocdl_path.h"

#include <stdlib.h>

#include "tsl/platform/path.h"

#if !defined(PLATFORM_GOOGLE) && TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif
#include "xla/tsl/platform/logging.h"

namespace tsl {

std::string RocmRoot() {
#if TENSORFLOW_USE_ROCM
  if (const char* rocm_path_env = std::getenv("ROCM_PATH")) {
    VLOG(3) << "ROCM root = " << rocm_path_env;
    return rocm_path_env;
  } else {
    VLOG(3) << "ROCM root = " << TF_ROCM_TOOLKIT_PATH;
    return TF_ROCM_TOOLKIT_PATH;
  }
#else
  return "";
#endif
}

std::string RocdlRoot() {
  if (const char* device_lib_path_env = std::getenv("HIP_DEVICE_LIB_PATH")) {
    return device_lib_path_env;
  } else {
    return io::JoinPath(RocmRoot(), "amdgcn/bitcode");
  }
}

}  // namespace tsl
