#ifndef TRITONGPU_IR_TYPES_H_
#define TRITONGPU_IR_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/Types.h.inc"

#include "triton/Dialect/TritonGPU/IR/TypeInterfaces.h.inc"

#endif // TRITON_IR_TYPES_H_
