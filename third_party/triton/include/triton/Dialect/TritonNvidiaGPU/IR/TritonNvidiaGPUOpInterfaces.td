#ifndef TRITON_NVIDIAGPU_OP_INTERFACES
#define TRITON_NVIDIAGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def MMAv5OpInterface : OpInterface<"MMAv5OpInterface"> {
  let description = [{
     This interface is implemented by MMAv5 dot and dot scaled ops.
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  // We can add more methods as needed.
  let methods = [
    InterfaceMethod<"Return the accumulator init flag.",
                    "::mlir::Value",
                    "useAccumulator">,
    InterfaceMethod<"Set the accumulator init flag.",
                    "void",
                    "setUseAccumulator",
                    (ins "::mlir::Value":$flag)>,
    InterfaceMethod<"Associate a new barrier to this MMAv5 op.",
                    "void",
                    "setBarrier",
                    (ins "::mlir::Value":$barrier)>,
    InterfaceMethod<"Return the accumulator.",
                    "::mlir::Value",
                    "getAccumulator">,
    InterfaceMethod<"Set the accumulator.",
                    "void",
                    "setAccumulator",
                    (ins "::mlir::Value":$accum)>,
    InterfaceMethod<"Return the predicate of this op.",
                    "::mlir::Value",
                    "getPredicate">,
    InterfaceMethod<"Set the predicate of this op.",
                    "void",
                    "setPredicate",
                    (ins "::mlir::Value":$pred)>,
  ];
}
#endif // TRITON_NVIDIAGPU_OP_INTERFACES
