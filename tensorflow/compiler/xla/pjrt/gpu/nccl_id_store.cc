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

#include "tensorflow/compiler/xla/pjrt/gpu/nccl_id_store.h"

#include <string>
#include <utility>

#ifdef NCCL_ENABLED
#if TENSORFLOW_USE_ROCM
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif
#else
#include "third_party/nccl/nccl.h"
#endif
#endif  // NCCL_ENABLED

#include "tensorflow/compiler/xla/util.h"

namespace xla {

StatusOr<std::string> NcclIdStore::GetNcclUniqueId(
    const gpu::NcclCliqueKey& key) {
  // The caller must ensure that threads calling this method concurrently have
  // unique keys, otherwise the global key-value store may hold the wrong value.
  {
    absl::MutexLock lock(&mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
  }
  std::string id_string;
  int primary_node_id = device_to_node_.at(key.devices()[0]);
  if (node_id_ == primary_node_id) {
#ifdef NCCL_ENABLED
    ncclUniqueId id;
    ncclResult_t r = ncclGetUniqueId(&id);
    TF_RET_CHECK(r == ncclSuccess);
    id_string = std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
    TF_RETURN_IF_ERROR(client_->KeyValueSet(key.ToString(), id_string));
#else
    return FailedPrecondition("NCCL support was not built into XLA binary.");
#endif
  } else {
    TF_ASSIGN_OR_RETURN(id_string, client_->BlockingKeyValueGet(
                                       key.ToString(), absl::Minutes(5)));
  }
  absl::MutexLock lock(&mu_);
  auto result = cache_.emplace(key, std::move(id_string));
  TF_RET_CHECK(result.second) << "Unique ID already in cache.";
  return result.first->second;
}

}  // namespace xla
