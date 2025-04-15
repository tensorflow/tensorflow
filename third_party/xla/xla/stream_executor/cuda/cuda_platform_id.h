/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_PLATFORM_ID_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_PLATFORM_ID_H_

#include "xla/stream_executor/platform.h"

namespace stream_executor {
namespace cuda {

// Opaque and unique identifier for the cuda platform.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a CudaPlatform object.
// This is broken out here to avoid a circular dependency between CudaPlatform
// and CudaExecutor.
extern const Platform::Id kCudaPlatformId;

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_PLATFORM_ID_H_
