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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_CUSTOM_CALL_IMPORTER_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_CUSTOM_CALL_IMPORTER_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

// Imports custom_calls prefixed with `mhlo.` from HLO to MHLO.
// This is used for ops in MHLO / StableHLO that don't exist in HLO. Many of
// these ops are needed for XlaBuilder clients that need to raise HLO to
// StableHLO.
absl::StatusOr<mlir::Operation*> ImportCustomCallAsOp(
    const HloCustomCallInstruction* instruction, mlir::Location loc,
    mlir::Type result_type, mlir::ValueRange operands,
    mlir::OpBuilder* builder);

// Indicates whether a custom call is an encoded MHLO op.
// Currently returns true for `mhlo.` prefixed custom calls.
bool IsOpEncodedCustomCall(const HloCustomCallInstruction* instruction);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_CUSTOM_CALL_IMPORTER_H_
