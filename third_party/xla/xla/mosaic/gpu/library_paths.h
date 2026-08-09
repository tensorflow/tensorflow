/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_GPU_LIBRARY_PATHS_H_
#define JAXLIB_MOSAIC_GPU_LIBRARY_PATHS_H_

#include <cstdlib>

namespace mosaic {
namespace gpu {

inline const char *GetCUDARoot() {
  return getenv("CUDA_ROOT");
}

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_LIBRARY_PATHS_H_
