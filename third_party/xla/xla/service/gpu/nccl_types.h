/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_NCCL_TYPES_H_
#define XLA_SERVICE_GPU_NCCL_TYPES_H_

#if XLA_ENABLE_XCCL
// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"  // IWYU pragma: export
#else
#include "rocm/include/rccl.h"  // IWYU pragma: export
#endif
#else
#include "third_party/nccl/nccl.h"
#endif
#endif  // XLA_ENABLE_XCCL

namespace xla::gpu {

#if defined(XLA_ENABLE_XCCL)

using NcclCommHandle = ncclComm_t;
using NcclDataType = ncclDataType_t;
using NcclRedOp = ncclRedOp_t;
using NcclStatus = ncclResult_t;
using NcclUniqueId = ncclUniqueId;

#else

// If we are compiling without NCCL support we define NCCL aliases as void
// pointers and always return errors in implementations. By doing this we can
// keep all XLA headers compilable even if NCCL is not available and do not
// spread ifdefs throughout the code base.
using NcclCommHandle = void*;
using NcclDataType = void*;
using NcclRedOp = void*;
using NcclStatus = void*;
using NcclUniqueId = void*;

#endif

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_NCCL_TYPES_H_
