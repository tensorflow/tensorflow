/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_RELOCATABLE_H_
#define XLA_PJRT_PJRT_RELOCATABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Represents a relocatable intermediate compilation artifact that can be
// called by other programs. Relocatables can be linked together into an
// executable. Relocatables are analogous to a object files and serve the
// same purpose of the allowing one to compile a subset of a program
// separately.
//
// Relocatables do not currently support ABI versioning, so users of
// relocatables must ensure that the same compiler is producing and
// consuming them.
class PjRtRelocatable {
 public:
  virtual ~PjRtRelocatable() = default;

  // Returns the name of this relocatable.
  virtual absl::string_view name() const = 0;

  // Returns the build ID of this relocatable, derived from the inputs that
  // produced it.
  virtual absl::string_view build_id() const = 0;

  // Returns the backend config to apply to any call sites for this
  // relocatable. The configuration is opaque and to be interpreted
  // by a given backend.
  virtual absl::string_view call_backend_config() const = 0;

  // Returns whether the relocatable has side effects.
  virtual bool has_side_effects() const = 0;

  // Serializes this executable into a string. The compatibility of the
  // serialized relocatable is implementation-specific.
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Returns a list of element types for each parameter.
  virtual absl::StatusOr<std::vector<PrimitiveType>> GetParameterElementTypes()
      const = 0;

  // Returns a list of element types for each output.
  virtual absl::StatusOr<std::vector<PrimitiveType>> GetOutputElementTypes()
      const = 0;

  // Returns a list of dimensions for each parameter.
  virtual absl::StatusOr<std::vector<DimensionVector>> GetParameterDimensions()
      const = 0;

  // Returns a list of dimensions for each output.
  virtual absl::StatusOr<std::vector<DimensionVector>> GetOutputDimensions()
      const = 0;

  // Returns the layout of each input parameter.
  virtual absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const = 0;

  // Returns the layout of each output.
  virtual absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const = 0;

  // Returns which pairs of parameters and outputs may alias.
  virtual absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>>
  GetParameterOutputAliases() const = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_RELOCATABLE_H_
