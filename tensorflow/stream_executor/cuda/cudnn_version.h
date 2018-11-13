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

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDNN_VERSION_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDNN_VERSION_H_

#include <string>

#include "tensorflow/core/lib/strings/strcat.h"

namespace stream_executor {
namespace cuda {

struct CudnnVersion {
  CudnnVersion() = default;

  CudnnVersion(int major, int minor, int patch)
      : major_version(major), minor_version(minor), patch_level(patch) {}

  tensorflow::string ToString() const {
    return tensorflow::strings::StrCat(major_version, ".", minor_version, ".",
                                       patch_level);
  }

  int major_version;
  int minor_version;
  int patch_level;
};

// Returns true if the given source CuDNN version is compatible with the given
// loaded version.
bool IsSourceCompatibleWithCudnnLibrary(CudnnVersion source_version,
                                        CudnnVersion loaded_version);

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDNN_VERSION_H_
