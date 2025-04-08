#ifndef TRITON_IR_OP_INTERFACES_H_
#define TRITON_IR_OP_INTERFACES_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {

namespace triton {

namespace impl {

LogicalResult verifyTransposeOpInterface(Operation *op);

LogicalResult verifyDotOpInterface(Operation *op);

} // namespace impl

} // namespace triton
} // namespace mlir

#include "triton/Dialect/Triton/IR/OpInterfaces.h.inc"

#endif // TRITON_IR_OP_INTERFACES_H_
