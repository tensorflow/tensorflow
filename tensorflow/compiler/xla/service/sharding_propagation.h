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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SHARDING_PROPAGATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SHARDING_PROPAGATION_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Propagates sharding information around the graph. HLOs that have shardings
// are kept as-is, those that do not have shardings are given shardings based on
// a simple local greedy heuristic.
class ShardingPropagation : public HloModulePass {
 public:
  explicit ShardingPropagation(bool is_spmd = false,
                               bool propagate_metadata = false)
      : is_spmd_(is_spmd), propagate_metadata_(propagate_metadata) {}
  absl::string_view name() const override { return "sharding-propagation"; }
  StatusOr<bool> Run(HloModule* module) override;

  // Function which can be used to apply a spatially partitioned sharding onto a
  // given domain. It will apply the sharding into the exit edges of the domain
  // and then rely on the rest of sharding propagation to ensure that the
  // intermediate nodes get the correct sharding.
  static Status NormalizeDomain(const DomainMetadata::Domain& domain,
                                const DomainMetadata* metadata);

 private:
  bool is_spmd_;
  bool propagate_metadata_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHARDING_PROPAGATION_H_
