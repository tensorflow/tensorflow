/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_QUANTIZATION_UNIT_LOC_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_QUANTIZATION_UNIT_LOC_H_

#include <optional>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir {
namespace quant {

// QuantizationUnitLoc uses CallSiteLoc as the base class so it can be printed
// with AsmPrinter and used to set the node name in MLIR to GraphDef exporter.
// The callee is named as `node_name@func_name` with child loc named as
// `op_type` while the caller is the quantization unit.
class QuantizationUnitLoc : public CallSiteLoc {
 public:
  using QuantizationUnit =
      tensorflow::quantization::UnitWiseQuantizationSpec::QuantizationUnit;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizationUnitLoc)

  QuantizationUnitLoc(MLIRContext* context, const QuantizationUnit& unit);

  // Checks if the given location is QuantizationUnitLoc. Users could call
  // `isa<QuantizationUnitLoc>(loc)` to check if the type matches.
  static bool classof(Attribute attr);
};

// Finds the QuantizationUnit from location info.
std::optional<QuantizationUnitLoc::QuantizationUnit>
FindQuantizationUnitFromLoc(Location loc);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_QUANTIZATION_UNIT_LOC_H_
