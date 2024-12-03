/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/nccl_id_store.h"

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<gpu::NcclCliqueId> NcclIdStore::GetNcclUniqueId(
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
  gpu::NcclCliqueId clique_id;
  int primary_node_id = device_to_node_.at(key.devices()[0]);
  if (node_id_ == primary_node_id) {
    TF_ASSIGN_OR_RETURN(clique_id, gpu::NcclApi::Default()->GetUniqueId());
    TF_RETURN_IF_ERROR(kv_store_->Set(key.ToString(), clique_id.ToString()));
  } else {
    TF_ASSIGN_OR_RETURN(std::string id_str,
                        kv_store_->Get(key.ToString(), absl::Minutes(10)));
    clique_id = gpu::NcclCliqueId(id_str);
  }
  absl::MutexLock lock(&mu_);
  auto result = cache_.emplace(key, std::move(clique_id));
  TF_RET_CHECK(result.second) << "Unique ID already in cache.";
  return result.first->second;
}

}  // namespace xla
