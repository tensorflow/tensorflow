/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPACE_TO_BATCH_CONVERTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPACE_TO_BATCH_CONVERTER_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

// Controller of various knobs.
struct SpaceToBatchController {
  bool enable_propagations_on_base_dilations;
  bool enable_propagations_on_window_dilations;
  bool enable_propagations_on_trivial_window_dilations;
  bool disable_starting_on_small_chains;
  int64 limit_on_batch_size;
};

// A pass which rewrites convolutions such that space dimension is turned into
// batch.
class SpaceToBatchConverter : public HloModulePass {
 public:
  explicit SpaceToBatchConverter(SpaceToBatchController ctrl) : ctrl_(ctrl) {}

  absl::string_view name() const override { return "space-to-batch-converter"; }

  // Run convolution rewriting on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

  // Controller for various knobs.
  SpaceToBatchController ctrl_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPACE_TO_BATCH_CONVERTER_H_
