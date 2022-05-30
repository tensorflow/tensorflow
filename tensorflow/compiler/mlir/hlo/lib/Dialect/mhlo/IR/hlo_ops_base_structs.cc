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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"

#include <sys/types.h>

#include <set>
#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

// Auto-generated implementation.
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.cc.inc"

namespace mlir {
namespace mhlo {

void printConvolutionDimensions(AsmPrinter &p, Operation *,
                                ConvDimensionNumbersAttr dnums_) {
  printConvolutionDimensions(p, dnums_);
}

}  // namespace mhlo
}  // namespace mlir
