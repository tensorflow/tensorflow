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
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<CliqueId> NcclIdStore::GetNcclUniqueId(const CliqueKey& key) {
  auto* gpu_key = tsl::down_cast<const gpu::GpuCliqueKey*>(&key);
  if (gpu_key == nullptr) {
    return InvalidArgument("Expected GPU clique key");
  }

  // The caller must ensure that threads calling this method concurrently have
  // unique keys, otherwise the global key-value store may hold the wrong value.
  {
    absl::MutexLock lock(&mu_);
    auto it = cache_.find(*gpu_key);
    if (it != cache_.end()) {
      return it->second;
    }
  }
  CliqueId clique_id;
  int primary_node_id = device_to_node_.at(gpu_key->devices()[0]);
  if (node_id_ == primary_node_id) {
    TF_ASSIGN_OR_RETURN(clique_id,
                        gpu::GpuCollectives::Default()->CreateUniqueCliqueId());
    TF_RETURN_IF_ERROR(
        kv_store_->Set(gpu_key->ToString(), clique_id.ToString()));
  } else {
    TF_ASSIGN_OR_RETURN(std::string id_str,
                        kv_store_->Get(gpu_key->ToString(), absl::Minutes(10)));
    clique_id = CliqueId(id_str);
  }
  absl::MutexLock lock(&mu_);
  auto result = cache_.emplace(*gpu_key, std::move(clique_id));
  TF_RET_CHECK(result.second) << "Unique ID already in cache.";
  return result.first->second;
}

}  // namespace xla
