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

#ifndef XLA_BACKENDS_CPU_ONEDNN_SUPPORT_H_
#define XLA_BACKENDS_CPU_ONEDNN_SUPPORT_H_

// oneDNN-fusion-related defines that don't depend on oneDNN Graph API.
// For anything dependent on Graph API, put it in onednn_fusion.h.

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

inline constexpr absl::string_view kOneDnnFusionKind = "__onednn_fusion";
using dnnl::graph::op;

bool IsOneDnnSupportedDType(PrimitiveType dtype,
                            const TargetMachineFeatures* cpu_features);

// Returns true if the shape doesn't have a layout or the layout is descending.
bool IsOneDnnSupportedLayout(const Shape& shape);

// Returns true if the HLO instruction and all its operands have supported data
// types and layouts.
bool IsOneDnnSupportedTypeAndLayout(
    const HloInstruction* hlo,
    const TargetMachineFeatures* cpu_features = nullptr);

// Returns true if the dot operation is supported by oneDNN. Returns an error
// if the dot operation shape is invalid.
absl::StatusOr<bool> IsDotSupportedByOneDnn(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape,
    const TargetMachineFeatures* cpu_features = nullptr);

// Returns the mappings from HLO opcodes to OneDNN unary operators.
const absl::flat_hash_map<HloOpcode, op::kind>& GetOneDnnUnaryOpMap();

// Returns the OneDNN unary operator corresponding to the given HLO opcode.
// Returns `InvalidArgument` if the opcode is not supported.
absl::StatusOr<op::kind> OneDnnUnaryOperator(const HloOpcode& opcode);

// Returns the names of the OneDNN supported HLO unary ops.
std::vector<absl::string_view> GetOneDnnSupportedUnaryOpsStrings();

// Returns the mappings from HLO opcodes to OneDNN binary operators.
const absl::flat_hash_map<HloOpcode, op::kind>& GetOneDnnBinaryOpMap();

// Returns the OneDNN binary operator corresponding to the given HLO opcode.
// Returns `InvalidArgument` if the opcode is not supported.
absl::StatusOr<op::kind> OneDnnBinaryOperator(const HloOpcode& opcode);

// Returns the names of the OneDNN supported HLO binary ops.
std::vector<absl::string_view> GetOneDnnSupportedBinaryOpsStrings();

// Returns true if the HLO op is supported by OneDNN.
bool IsOpSupportedByOneDnn(const HloInstruction* hlo,
                           const TargetMachineFeatures* cpu_features = nullptr);

// Returns true if the constant is supported by OneDNN.
bool IsConstantSupportedByOneDnn(
    const HloInstruction* hlo,
    const TargetMachineFeatures* cpu_features = nullptr);

// Returns true if the bitcast op is supported by OneDNN.
bool IsBitcastOpSupportedByOneDnn(
    const HloInstruction* hlo,
    const TargetMachineFeatures* cpu_features = nullptr);

// Returns true if the elementwise op is supported by OneDNN.
bool IsElementwiseOpSupportedByOneDnn(
    const HloInstruction* hlo,
    const TargetMachineFeatures* cpu_features = nullptr);
}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_ONEDNN_SUPPORT_H_
