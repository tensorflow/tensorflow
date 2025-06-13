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

#include "tsl/platform/cuda_root_path.h"

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
#include "xla/tsl/platform/env.h"
#endif
#include "xla/tsl/platform/logging.h"

namespace tsl {

std::vector<std::string> CandidateCudaRoots() {
#if !defined(PLATFORM_GOOGLE)
  auto roots = std::vector<std::string>{};
  std::string runfiles_suffix = "runfiles";
  std::vector<std::string> cuda_dir_names = {"cuda_nvcc", "cuda_nvdisasm",
                                             "nvidia_nvshmem"};

  // The CUDA candidate root for c++ targets.
  std::string executable_path = tsl::Env::Default()->GetExecutablePath();
  for (const std::string& cuda_dir_name : cuda_dir_names) {
    std::string cuda_dir =
        io::JoinPath(executable_path + "." + runfiles_suffix, cuda_dir_name);
    roots.push_back(cuda_dir);
  }

  // The CUDA candidate root for python targets.
  std::string runfiles_dir = tsl::Env::Default()->GetRunfilesDir();
  std::size_t runfiles_ind = runfiles_dir.rfind(runfiles_suffix);
  for (const std::string& cuda_dir_name : cuda_dir_names) {
    std::string cuda_dir = io::JoinPath(
        runfiles_dir.substr(0, runfiles_ind + runfiles_suffix.length()),
        cuda_dir_name);
    roots.push_back(cuda_dir);
  }

  roots.push_back(TF_CUDA_TOOLKIT_PATH);
  roots.emplace_back(std::string("/usr/local/cuda"));
  roots.emplace_back(std::string("/opt/cuda"));

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

    // Also add the path to the copy of libdevice.10.bc that we include within
    // the Python wheel.
    roots.emplace_back(io::JoinPath(dir, "cuda"));

    // In case cuda was installed with nvidia's official conda packages, we also
    // include the root prefix of the environment in the candidate roots dir,
    // we assume that the lib binaries are either in the python package's root
    // dir or in a 'python' subdirectory, as done by the previous for. python
    // packages on non-Windows platforms are installed in
    // $CONDA_PREFIX/lib/python3.12/site-packages/pkg_name, so if we want
    // to add $CONDA_PREFIX to the candidate roots dirs we need to add
    // ../../../..
    for (auto path : {"../../../..", "../../../../.."})
      roots.emplace_back(io::JoinPath(dir, path));
  }
#endif  // defined(PLATFORM_POSIX) && !defined(__APPLE__)

  for (auto root : roots) VLOG(3) << "CUDA root = " << root;
  return roots;
#else   // !defined(PLATFORM_GOOGLE)
  return {};
#endif  //! defined(PLATFORM_GOOGLE)
}

bool PreferPtxasFromPath() { return true; }

}  // namespace tsl
