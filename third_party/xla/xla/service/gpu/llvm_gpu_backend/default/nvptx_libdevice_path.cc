
/* Copyright 2024 The OpenXLA Authors.
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
#include <string>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_utils.h"
#include "tsl/platform/cuda_root_path.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"

namespace xla::gpu::nvptx {
namespace {

std::string GetLibdeviceDir(absl::string_view xla_gpu_cuda_data_dir) {
  for (const std::string& cuda_root :
       tsl::CandidateCudaRoots(std::string{xla_gpu_cuda_data_dir})) {
    std::string libdevice_dir =
        tsl::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tsl::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  LOG(WARNING) << CantFindCudaMessage(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may "
      "result in compilation or runtime failures, if the program we try to run "
      "uses routines from libdevice.",
      xla_gpu_cuda_data_dir);
  // GetCudaRootCandidates always includes ".", but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
}

}  // namespace

std::string LibDevicePath(absl::string_view xla_gpu_cuda_data_dir) {
  static absl::Mutex libdevice_cache_mu(absl::kConstInit);
  static auto& libdevice_dir_path_cache ABSL_GUARDED_BY(libdevice_cache_mu) =
      *new absl::flat_hash_map<std::string, std::string>();
  std::string libdevice_dir_path = [&] {
    absl::MutexLock l(&libdevice_cache_mu);
    auto it = libdevice_dir_path_cache.find(xla_gpu_cuda_data_dir);
    if (it != libdevice_dir_path_cache.end()) {
      return it->second;
    }
    auto [it2, inserted] = libdevice_dir_path_cache.emplace(
        xla_gpu_cuda_data_dir, GetLibdeviceDir(xla_gpu_cuda_data_dir));
    return it2->second;
  }();
  // CUDA 9+ uses a single libdevice file for all devices, and we don't support
  // older CUDAs.
  return tsl::io::JoinPath(libdevice_dir_path, "libdevice.10.bc");
}

}  // namespace xla::gpu::nvptx
