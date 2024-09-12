/* Copyright 2017 The OpenXLA Authors.

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

// Utilities to manipulate data in hlo.proto.

#ifndef XLA_SERVICE_HLO_PROTO_UTIL_H_
#define XLA_SERVICE_HLO_PROTO_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo.pb.h"

namespace xla {

// Returns a serialized representation of the HLO state.
HloProto MakeHloProto(const HloModule& module,
                      const BufferAssignment& assignment);

// Returns a serialized representation of the HLO state, but buffer assignment
// will not be included in the output.
HloProto MakeHloProto(const HloModule& module);

// Create an HLO state from serialized representation. In addition to
// creating the proto with HloModule::CreateFromProto(...) it also
// uses HloVerifier to ensure basic invariants are held.
// The HLO module could be a pre-optimizations (default) or post-optimizations
// module, which affects how the HLO module is verified, e.g., mixed-precision
// is allowed in post-optimizations HLOs.
absl::StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    bool is_module_post_optimizations = false);

// Returns the shapes of the parameters of the entry computation. Shape pointers
// refer to shapes inside of the given HloProto.
absl::StatusOr<std::vector<const ShapeProto*>> EntryComputationParameterShapes(
    const HloProto& hlo_proto);

// Returns the shape of the output of the entry computation. The shape pointer
// refers to the output shape inside of the given HloProto.
absl::StatusOr<const ShapeProto*> EntryComputationOutputShape(
    const HloProto& hlo_proto);

}  // namespace xla

#endif  // XLA_SERVICE_HLO_PROTO_UTIL_H_
