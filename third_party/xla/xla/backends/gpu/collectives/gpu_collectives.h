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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_

#include <functional>

#include "absl/status/statusor.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"

namespace xla::gpu {

// XLA:GPU extension of the Collectives interface with GPU-specific APIs.
class GpuCollectives : public Collectives {
 public:
  // A callback to get a unique clique id.
  using CliqueIdCallback =  // NOLINT
      std::function<absl::StatusOr<CliqueId>(const CliqueKey&)>;

  // Returns true if collectives backend uses global config.
  virtual bool IsGlobalConfig() const = 0;

  // Returns a clique id callback passed as an argument if it's not null or a
  // default callback to get create a clique id if we are running in local mode.
  virtual absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback* clique_id_callback, bool is_local) = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_
