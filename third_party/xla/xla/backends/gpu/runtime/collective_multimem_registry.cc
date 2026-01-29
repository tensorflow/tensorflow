/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_multimem_registry.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/collective_multimem.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

void CollectiveMultimemRegistry::Request(const MultimemRequest& request) {
  requests_.push_back(request);
}

absl::Status CollectiveMultimemRegistry::Build() {
  for (const MultimemRequest& request : requests_) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<CollectiveMultimem> multimem,
        CollectiveMultimem::Allocate(executor_, request.key, global_device_id_,
                                     request.map_to));
    multimems_[request] = multimem;
  }

  requests_.clear();
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CollectiveMultimem>>
CollectiveMultimemRegistry::Get(const MultimemRequest& request) const {
  auto it = multimems_.find(request);
  if (it == multimems_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Multimem not found for request: %s", request.key.ToString()));
  }
  return it->second;
}

}  // namespace xla::gpu
