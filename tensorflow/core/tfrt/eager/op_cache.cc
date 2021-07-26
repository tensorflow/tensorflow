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
#include "tensorflow/core/tfrt/eager/op_cache.h"

#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

Expected<CoreRuntimeOp*> OpCache::GetOrAddOp(
    string_view op_name, OpHandler* op_handler, string_view device_name,
    llvm::SmallVector<string_view, 4> dtypes,
    OperationInterface* const op_interface) {
  CacheKey cache_key{op_name, op_handler,
                     (op_handler == nullptr ? device_name : ""), dtypes};
  {
    mutex_lock l(cache_mu_);
    auto iter = cache_.find(cache_key);
    if (iter != cache_.end()) return &iter->second;
  }

  ContextInterface* context = op_interface->context_;

  auto tfrt_op_name = StrCat("tf.", op_name);
  op_interface->MaybeInferInputAttrs();
  if (op_handler == nullptr) {
    tensorflow::Status s = context->SelectOpHandlerFromNodeDef(
        *op_interface, &op_interface->fallback_attrs_.BuildNodeDef(),
        &op_handler);
    if (!s.ok()) return MakeStringError(s.error_message());
  }
  Expected<CoreRuntimeOp> expected_op =
      context->GetCoreRuntime()->MakeOp(tfrt_op_name, op_handler);
  if (!expected_op) return MakeStringError(expected_op.takeError());

  mutex_lock l(cache_mu_);
  // Insert the new op to cache. If an entry with the same key is already
  // present in the cache at this moment due to race condition, overwrites it.
  cache_key.MakeConcrete();
  cache_[cache_key] = std::move(expected_op.get());
  return &cache_[cache_key];
}

Expected<CoreRuntimeOp*> OpCache::GetOrAddXlaOp(string_view op_name,
                                                ContextInterface* context) {
  // Device name and dtype are not meaningful to a XLA op.
  CacheKey cache_key{op_name, nullptr, "", {}};
  {
    mutex_lock l(cache_mu_);
    auto iter = cache_.find(cache_key);
    if (iter != cache_.end()) return &iter->second;
  }

  auto tfrt_op_name = StrCat("tf.", op_name);
  Expected<CoreRuntimeOp> expected_op = context->GetCoreRuntime()->MakeOp(
      tfrt_op_name, context->GetFallbackOpHandler());
  if (!expected_op) return MakeStringError(expected_op.takeError());

  mutex_lock l(cache_mu_);
  // Insert the new op to cache. If an entry with the same key is already
  // present in the cache at this moment due to race condition, overwrites it.
  cache_key.MakeConcrete();
  cache_[cache_key] = std::move(expected_op.get());
  return &cache_[cache_key];
}

}  // namespace tf
}  // namespace tfrt
