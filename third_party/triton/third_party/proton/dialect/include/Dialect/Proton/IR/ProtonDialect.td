#ifndef PROTON_DIALECT
#define PROTON_DIALECT

include "mlir/IR/OpBase.td"

def Proton_Dialect : Dialect {
  let name = "proton";
  let cppNamespace = "::mlir::triton::proton";

  let description = [{
    Proton Dialect provides core ops for building third-party compiler-based
    performance profiling and analysis tools.
  }];

  let dependentDialects = [];
}

#endif
