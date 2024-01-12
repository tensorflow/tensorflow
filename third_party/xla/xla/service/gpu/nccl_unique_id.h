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

#ifndef XLA_SERVICE_GPU_NCCL_UNIQUE_ID_H_
#define XLA_SERVICE_GPU_NCCL_UNIQUE_ID_H_

#include <functional>
#include <string>

#include "absl/status/statusor.h"
namespace xla::gpu {

// Forward declare type defined in nccl_clique.h.
class NcclCliqueKey;

// Returns true if the NCCL config is global (NCCL_COMM_ID env variable is set).
bool IsGlobalNcclConfig();

//===----------------------------------------------------------------------===//
// NcclUniqueId
//===----------------------------------------------------------------------===//

// A callback to get a unique nccl clique id (see `ncclUniqueId` documentation).
using NcclUniqueIdCallback =  // NOLINT
    std::function<absl::StatusOr<std::string>(const NcclCliqueKey&)>;

// Returns a unique id callback passed as an argument if it's not null or a
// default callback to get NCCL id if we are running in local mode.
absl::StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback,  // may be null
    bool is_local);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_NCCL_UNIQUE_ID_H_
