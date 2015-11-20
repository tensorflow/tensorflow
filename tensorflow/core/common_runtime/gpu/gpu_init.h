/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_INIT_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_INIT_H_

namespace perftools {
namespace gputools {
class Platform;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

// Returns the GPU machine manager singleton, creating it and
// initializing the GPUs on the machine if needed the first time it is
// called.
perftools::gputools::Platform* GPUMachineManager();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_INIT_H_
