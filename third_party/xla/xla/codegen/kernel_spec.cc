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

#include "xla/codegen/kernel_spec.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "xla/stream_executor/launch_dim.h"

namespace xla {

KernelSpec::KernelSpec(se::ClusterDim cluster_dim, se::BlockDim block_dim,
                       se::ThreadDim thread_dim,
                       std::optional<size_t> scratch_bytes,
                       BufferUses buffer_uses)
    : cluster_dim_(cluster_dim),
      block_dim_(block_dim),
      thread_dim_(thread_dim),
      scratch_bytes_(scratch_bytes),
      buffer_uses_(std::move(buffer_uses)) {}

}  // namespace xla
