#ifndef TRITON_DIALECT_PROTON_IR_DIALECT_H_
#define TRITON_DIALECT_PROTON_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h.inc"
#include "proton/dialect/include/Dialect/Proton/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "proton/dialect/include/Dialect/Proton/IR/ProtonAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "proton/dialect/include/Dialect/Proton/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace proton {} // namespace proton
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_PROTON_IR_DIALECT_H_
