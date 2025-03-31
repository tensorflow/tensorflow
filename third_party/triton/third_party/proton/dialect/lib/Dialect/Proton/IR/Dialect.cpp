#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/Proton/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::proton;

void mlir::triton::proton::ProtonDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Proton/IR/ProtonAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Proton/IR/Ops.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/Proton/IR/ProtonAttrDefs.cpp.inc"
