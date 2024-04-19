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

#ifndef XLA_SERVICE_SUB_BYTE_NORMALIZATION_H_
#define XLA_SERVICE_SUB_BYTE_NORMALIZATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// A pass that can modify the sub-byte element_size_in_bits annotation on
// layouts. Depending on the constructor argument, it either removes the
// element_size_in_bits annotation for platforms that doesn't support
// nibble-packed types, or it sets element_size_in_bits to 4 for 4-bit values.
class SubByteNormalization : public HloModulePass {
 public:
  enum Mode {
    // Remove element_size_in_bits on all layouts. Useful for platforms which
    // do not support nibble-packed types.
    REMOVE_ELEMENT_SIZE,
    // Set element_size_in_bits to 4 for layouts of int4 types (S4, U4), and to
    // 0 for all other layouts. Useful for platforms which support nibble-packed
    // types.
    SET_ELEMENT_SIZE,
  };

  explicit SubByteNormalization(Mode mode) : mode_(mode) {}

  ~SubByteNormalization() override = default;

  absl::string_view name() const override {
    switch (mode_) {
      case REMOVE_ELEMENT_SIZE:
        return "int4-size-removal";
      case SET_ELEMENT_SIZE:
        return "int4-size-setter";
    }
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Mode mode_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SUB_BYTE_NORMALIZATION_H_
