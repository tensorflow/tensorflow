/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/cuda_libdevice_path.h"

#include <stdlib.h>

#include <string>
#include <vector>

#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"

#if defined(PLATFORM_POSIX) && !defined(__APPLE__)
#include <dlfcn.h>
#include <libgen.h>
#include <link.h>
#endif

#if !defined(PLATFORM_GOOGLE)
#include "third_party/gpus/cuda/cuda_config.h"
#endif
#include "tsl/platform/logging.h"

namespace tsl {

std::vector<std::string> CandidateCudaRoots() {
#if !defined(PLATFORM_GOOGLE)
  auto roots = std::vector<std::string>{TF_CUDA_TOOLKIT_PATH,
                                        std::string("/usr/local/cuda")};

#if defined(PLATFORM_POSIX) && !defined(__APPLE__)
  Dl_info info;

  if (dladdr(&__FUNCTION__, &info)) {
    auto lib = std::string(info.dli_fname);
    auto dir = io::Dirname(lib);

    // TF lib binaries are located in both the package's root dir and within a
    // 'python' subdirectory (for pywrap libs). So we check two possible paths
    // relative to the current binary for the wheel-based nvcc package.
    for (auto path : {"../nvidia/cuda_nvcc", "../../nvidia/cuda_nvcc"})
      roots.emplace_back(io::JoinPath(dir, path));
  }
#endif  // defined(PLATFORM_POSIX) && !defined(__APPLE__)

  for (auto root : roots) VLOG(3) << "CUDA root = " << root;
  return roots;
#else   // !defined(PLATFORM_GOOGLE)
  return {std::string("/usr/local/cuda")};
#endif  //! defined(PLATFORM_GOOGLE)
}

bool PreferPtxasFromPath() { return true; }

}  // namespace tsl
