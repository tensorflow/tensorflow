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

#ifndef XLA_SERVICE_GPU_NCCL_API_H_
#define XLA_SERVICE_GPU_NCCL_API_H_

#include "absl/status/statusor.h"
#include "xla/service/gpu/nccl_clique_key.h"

namespace xla::gpu {

struct NcclApi {
  // Creates a new unique clique id.
  //
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclgetuniqueid
  static absl::StatusOr<NcclCliqueId> GetUniqueId();
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_NCCL_API_H_
