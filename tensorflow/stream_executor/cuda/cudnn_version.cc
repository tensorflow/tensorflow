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

#include "tensorflow/stream_executor/cuda/cudnn_version.h"

namespace stream_executor {
namespace gpu {

bool IsSourceCompatibleWithCudnnLibrary(CudnnVersion source_version,
                                        CudnnVersion loaded_version) {
  // Major version is neither forward or backward compatible and therefore major
  // versions needs to match between source and library.
  //
  // Minor version is backward-compatible and therefore minor version of library
  // needs to be same or higher.
  //
  // Patch releases are always forward and backward compatible and therefore
  // need not match.
  return loaded_version.major_version == source_version.major_version &&
         loaded_version.minor_version >= source_version.minor_version;
}

}  // namespace gpu
}  // namespace stream_executor
