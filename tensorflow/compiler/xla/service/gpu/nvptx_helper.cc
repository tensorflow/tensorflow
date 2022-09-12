/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nvptx_helper.h"

#include <string>

#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/tsl/platform/cuda_libdevice_path.h"

namespace xla {
namespace gpu {

namespace {

std::vector<std::string> CandidateCudaRoots(const HloModuleConfig& config) {
  return tsl::CandidateCudaRoots(
      config.debug_options().xla_gpu_cuda_data_dir());
}

}  // namespace

std::string CantFindCudaMessage(absl::string_view msg,
                                const HloModuleConfig& hlo_module_config) {
  return absl::StrCat(
      msg, "\nSearched for CUDA in the following directories:\n  ",
      absl::StrJoin(CandidateCudaRoots(hlo_module_config), "\n  "),
      "\nYou can choose the search directory by setting xla_gpu_cuda_data_dir "
      "in HloModule's DebugOptions.  For most apps, setting the environment "
      "variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.");
}

std::string GetLibdeviceDir(const HloModuleConfig& hlo_module_config) {
  for (const std::string& cuda_root : CandidateCudaRoots(hlo_module_config)) {
    std::string libdevice_dir =
        tsl::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tensorflow::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  LOG(WARNING) << CantFindCudaMessage(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may "
      "result in compilation or runtime failures, if the program we try to run "
      "uses routines from libdevice.",
      hlo_module_config);

  // GetCudaRootCandidates always includes ".", but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
}

}  // namespace gpu
}  // namespace xla
