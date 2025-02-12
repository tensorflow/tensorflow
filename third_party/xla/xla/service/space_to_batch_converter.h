/* Copyright 2018 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_SPACE_TO_BATCH_CONVERTER_H_
#define XLA_SERVICE_SPACE_TO_BATCH_CONVERTER_H_

#include <stdbool.h>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/status_macros.h"

namespace xla {

// Controller of various knobs.
struct SpaceToBatchController {
  bool enable_propagations_on_base_dilations;
  bool enable_propagations_on_window_dilations;
  bool enable_propagations_on_trivial_window_dilations;
  bool disable_starting_on_small_chains;
  int64_t limit_on_batch_size;
  int64_t dimension_from_end_to_convert = 1;
  // We choose the new batch size to be number_of_splits times that of the old
  // batch so that space-to-batch propagation through several convolutional
  // layers is consistent.
  int64_t number_of_splits = 8;
  int64_t count_of_dimensions_to_convert = 1;
};

// Represents the different dimension mappings. Can be extended as needed.
enum class SpaceToBatchDimMap : uint8_t {
  kBatch = 0,
  kFeature = 1,
  kSpace0 = 2,
};

// A pass which rewrites convolutions such that space dimension is turned into
// batch.
class SpaceToBatchConverter : public HloModulePass {
 public:
  explicit SpaceToBatchConverter(SpaceToBatchController ctrl) : ctrl_(ctrl) {}

  absl::string_view name() const override { return "space-to-batch-converter"; }

  // Run convolution rewriting on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Controller for various knobs.
  SpaceToBatchController ctrl_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SPACE_TO_BATCH_CONVERTER_H_
