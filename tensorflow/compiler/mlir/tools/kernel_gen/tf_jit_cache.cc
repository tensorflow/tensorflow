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

#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_jit_cache.h"

#include <functional>
#include <string>
#include <utility>

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

tensorflow::Status JITCache::Create(JITCache** dst) {
  *dst = new JITCache;
  return tensorflow::Status::OK();
}

std::string JITCache::DebugString() const { return "JIT cache"; }

ExecutionEngine* JITCache::LookupOrCompile(
    const std::string code,
    std::function<llvm::Expected<std::unique_ptr<ExecutionEngine>>()>
        compile_callback) {
  // Check if we already have a compiled module in the cache.
  {
    tensorflow::mutex_lock lock(mu_);
    if (execution_engine_by_key_.contains(code))
      return execution_engine_by_key_[code].get();
  }

  // Otherwise, compile the module now.
  llvm::Expected<std::unique_ptr<ExecutionEngine>> engine = compile_callback();
  if (!engine) return nullptr;

  // Insert the compiled module into our cache and return a raw pointer.
  {
    tensorflow::mutex_lock lock(mu_);
    // Check again whether we already have a compiled module in the cache. It
    // may have been added during the time we ran compile_callback().
    return execution_engine_by_key_.try_emplace(code, std::move(engine.get()))
        .first->second.get();
  }
}

size_t JITCache::Size() {
  tensorflow::mutex_lock lock(mu_);
  return execution_engine_by_key_.size();
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
