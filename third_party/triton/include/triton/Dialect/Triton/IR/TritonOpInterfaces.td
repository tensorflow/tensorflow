#ifndef TRITON_OP_INTERFACES
#define TRITON_OP_INTERFACES

include "mlir/IR/OpBase.td"


def TransposeOpInterface : OpInterface<"TransposeOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a transpose.
    It provides methods to access common properties such as the order attribute
    and the source operand.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the source operand of the transposition.",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Get the order of the transposition.",
      /*retType=*/"::mlir::ArrayRef<int32_t>",
      /*methodName=*/"getOrder",
      /*args=*/(ins)>
  ];

  let verify = [{
    return ::mlir::triton::impl::verifyTransposeOpInterface($_op);
  }];
}

def DotOpInterface : OpInterface<"DotOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a dot product.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the LHS A tensor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getA",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Get the RHS B tensor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getB",
      /*args=*/(ins)>,
  InterfaceMethod<
      /*desc=*/"Verify the dimensions of the A and B DotOp operands.",
      /*retType=*/"bool",
      /*methodName=*/"verifyDims",
      /*args=*/(ins)>
  ];

  let verify = [{ return ::mlir::triton::impl::verifyDotOpInterface($_op); }];
}


#endif // TRITON_OP_INTERFACES
