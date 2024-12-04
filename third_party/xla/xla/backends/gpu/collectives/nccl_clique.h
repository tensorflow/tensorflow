/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_CLIQUE_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_CLIQUE_H_

#include <string>

#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"

namespace xla::gpu {

// A GPU clique that is implemented on top of NCCL communicators.
//
// TODO(b/380457503): Remove `Impl` suffix once we migrate all users to
// LockableGpuClique.
class NcclCliqueImpl : public GpuClique {
 public:
  using GpuClique::GpuClique;

  std::string DebugString() const final;
  absl::Status HealthCheck() const final;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_CLIQUE_H_
