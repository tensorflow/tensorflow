/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_OP_CONSTS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_OP_CONSTS_H_

#include "absl/base/attributes.h"

namespace tensorflow {
namespace tpu {

// Resource names in the ResourceMgr.
//
// Name of cache for compiled TPU ISA protos. CompilationCache is created by
// ConfigureDistributedTpuOp, so only the master has a CompilationCache.
ABSL_CONST_INIT extern const char kCompilationCacheResourceName[];
// Name of base class allowing Execute Ops to look up ISA protos.
// CompiledProtoCache is created by InitializeHostForDistributedTpuOp, so each
// tpu_worker has a CompiledProtoCache.
ABSL_CONST_INIT extern const char kCompiledProtoCacheResourceName[];
// Name of cache unloader for compiled TPU ISA protos. Cache unloader should be
// put into TPU_SYSTEM device resource manager. Inference may use it to unload
// cache entries created during lifetime of a DirectSession.
ABSL_CONST_INIT extern const char kCompilationCacheUnloaderResourceName[];
// TBD
ABSL_CONST_INIT extern const char kFingerprintLookupResourceName[];

}  // namespace tpu
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_OP_CONSTS_H_
