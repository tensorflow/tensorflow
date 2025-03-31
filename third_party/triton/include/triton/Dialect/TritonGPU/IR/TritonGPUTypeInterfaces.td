#ifndef TRITON_GPU_TYPE_INTERFACES
#define TRITON_GPU_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

// Interface dynamically attached to RankedTensorType and MemDescType.
def TTG_TensorOrMemDesc : TypeInterface<"TensorOrMemDesc"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Returns the encoding of the tensor or memory descriptor",
      "mlir::Attribute", "getEncoding", (ins)>,
    InterfaceMethod<"Returns element type",
      "mlir::Type", "getElementType", (ins)>,
    InterfaceMethod<"Returns the type shape",
      "llvm::ArrayRef<int64_t>", "getShape", (ins)>,
    InterfaceMethod<"Returns the tensor or buffer rank",
      "int64_t", "getRank", (ins)>,
    InterfaceMethod<"Returns the element type bit width",
      "int64_t", "getElementTypeBitWidth", (ins)>,
  ];
}

#endif // TRITON_GPU_TYPE_INTERFACES
