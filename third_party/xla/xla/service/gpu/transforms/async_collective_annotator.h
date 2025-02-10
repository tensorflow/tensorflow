/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_ASYNC_COLLECTIVE_ANNOTATOR_H_
#define XLA_SERVICE_GPU_TRANSFORMS_ASYNC_COLLECTIVE_ANNOTATOR_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Annotate async collectives with CollectiveBackendConfig.
class AsyncCollectiveAnnotator : public HloModulePass {
 public:
  explicit AsyncCollectiveAnnotator(HloPredicate is_collective_async)
      : is_collective_async_(std::move(is_collective_async)) {}
  absl::string_view name() const override {
    return "async-collective-annotator";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloPredicate is_collective_async_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_ASYNC_COLLECTIVE_ANNOTATOR_H_
