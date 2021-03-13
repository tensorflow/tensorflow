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

#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_factory.h"

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_external.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"

namespace tensorflow {
namespace tpu {
namespace {

TpuCompilationCacheInterface* CreateCompilationCacheExternal() {
  // NOTE: Change the 1 << 33 value to change the compilation cache size.
  // TODO(frankchn): Make this configurable.
  return new TpuCompilationCacheExternal(int64{1} << 33);  // 8 GB
}

// Using a pointer here to fulfill the trivially destructible requirement for
// static variables.
static std::function<TpuCompilationCacheInterface*()>*
    compilation_cache_creation_fn =
        new std::function<TpuCompilationCacheInterface*()>(
            CreateCompilationCacheExternal);

}  // namespace

std::function<TpuCompilationCacheInterface*()> GetCompilationCacheCreateFn() {
  return *compilation_cache_creation_fn;
}

void SetCompilationCacheCreateFn(
    std::function<TpuCompilationCacheInterface*()> fn) {
  delete compilation_cache_creation_fn;
  compilation_cache_creation_fn =
      new std::function<TpuCompilationCacheInterface*()>(fn);
}

}  // namespace tpu
}  // namespace tensorflow
