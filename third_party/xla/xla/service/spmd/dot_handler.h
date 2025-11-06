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

#ifndef XLA_SERVICE_SPMD_DOT_HANDLER_H_
#define XLA_SERVICE_SPMD_DOT_HANDLER_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

class CreateShardedConvolutionFunctor;
class CreateShardedDotFunctor;
class CreateShardedScaledDotFunctor;

// Abstract base class for functors creating sharded dots, block-scaled dots and
// convolutions.
template <typename PartitionedHloMaybeMX>
class CreateShardedFunctorBase {
 public:
  virtual ~CreateShardedFunctorBase() = default;

  // Implemented in derived classes to create sharded dots, block-scaled dots
  // and convolutions.
  virtual absl::StatusOr<HloInstruction*> CreateSharded(
      const PartitionedHloMaybeMX& ll, const PartitionedHloMaybeMX& rr,
      SpmdBuilder* b, const Window& conv_window) const = 0;

  void SetCustomCreateSharded(
      std::function<absl::StatusOr<HloInstruction*>(
          const PartitionedHloMaybeMX&, const PartitionedHloMaybeMX&,
          SpmdBuilder*, const Window&)>&& custom_create_sharded) {
    custom_create_sharded_ = std::move(custom_create_sharded);
  }

  absl::StatusOr<HloInstruction*> operator()(
      const PartitionedHloMaybeMX& ll, const PartitionedHloMaybeMX& rr,
      SpmdBuilder* builder, const Window& conv_window,
      bool call_custom_create_sharded = true) const {
    if (call_custom_create_sharded && custom_create_sharded_) {
      return custom_create_sharded_(ll, rr, builder, conv_window);
    }
    return CreateSharded(ll, rr, builder, conv_window);
  }

 private:
  // May hold a function which can be optionally called instead of
  // CreateSharded.
  std::function<absl::StatusOr<HloInstruction*>(const PartitionedHloMaybeMX&,
                                                const PartitionedHloMaybeMX&,
                                                SpmdBuilder*, const Window&)>
      custom_create_sharded_;
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_DOT_HANDLER_H_
