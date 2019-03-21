// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def I32OrF32 : Type<CPred<"{0}.isInteger(32) || {0}.isF32()">,
                    "32-bit integer or floating-point type">;
def I32OrF32Tensor : TypedTensor<I32OrF32>;

def Identity : Op<"identity", []> {
  let arguments = (ins
      I32OrF32Tensor:$x);
  let results = (outs
      I32OrF32Tensor:$y);
}

// CHECK: this->getInstruction()->getOperand(0)->getType().cast<TensorType>().getElementType().isInteger(32) ||
// CHECK-SAME: this->getInstruction()->getOperand(0)->getType().cast<TensorType>().getElementType().isF32()
