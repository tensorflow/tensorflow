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

// This file defines the operations used in the LMHLO GPU dialect.

#include "lhlo_gpu/IR/lhlo_gpu_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "lhlo/utils/lhlo_utils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/IR/hlo_ops_common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace lmhlo_gpu {
static FailureOr<bool> parseBool(AsmParser &parser) {
  if (succeeded(parser.parseOptionalKeyword("true"))) return true;
  if (succeeded(parser.parseOptionalKeyword("false"))) return false;
  return failure();
}

static FailureOr<SmallVector<int64_t>> parseI64Array(AsmParser &parser) {
  SmallVector<int64_t> elements;
  auto elementParser = [&]() {
    int64_t element = 0;
    if (failed(parser.parseInteger(element))) return failure();
    elements.push_back(element);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     elementParser))
    return failure();
  return elements;
}
}  // namespace lmhlo_gpu
}  // namespace mlir

// Include order below matters.
#include "lhlo_gpu/IR/lhlo_gpu_ops_dialect.cc.inc"
#include "lhlo_gpu/IR/lhlo_gpu_ops_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "lhlo_gpu/IR/lhlo_gpu_ops_attrdefs.cc.inc"

namespace mlir {
namespace lmhlo_gpu {

using mhlo::TokenType;

void LmhloGpuDialect::initialize() {
  getContext()->loadDialect<mhlo::MhloDialect>();
  addOperations<
#define GET_OP_LIST
#include "lhlo_gpu/IR/lhlo_gpu_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lhlo_gpu/IR/lhlo_gpu_ops_attrdefs.cc.inc"
      >();
}

// TODO(jurahul): Add verification for operand shapes and ranks.

using mlir::hlo::parseWindowAttributes;
using mlir::hlo::printWindowAttributes;

//===----------------------------------------------------------------------===//
// AllReduceStartOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AllReduceStartOp::verify() {
  AllReduceStartOp op = *this;
  return lmhlo::verifyAllReduce(op);
}

}  // namespace lmhlo_gpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "lhlo_gpu/IR/lhlo_gpu_ops.cc.inc"
