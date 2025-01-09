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

#include <memory>

#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"

namespace xla {

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(pjrt_register_se_gpu_compiler, {
  PjRtRegisterCompiler(CudaName(), std::make_unique<StreamExecutorGpuCompiler>(
                                       stream_executor::cuda::kCudaPlatformId));
});

}  // namespace xla
