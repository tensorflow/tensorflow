// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def I32OrF32 : Type<CPred<"$_self.isInteger(32) || $_self.isF32()">,
                    "32-bit integer or floating-point type">;

def OpA : NS_Op<"op_for_CPred_containing_multiple_same_placeholder", []> {
  let arguments = (ins I32OrF32:$x);
}

// CHECK-LABEL: OpA::verify
// CHECK: if (!((this->getOperation()->getOperand(0)->getType().isInteger(32) || this->getOperation()->getOperand(0)->getType().isF32())))

def OpB : NS_Op<"op_for_AllOf_PredOpTrait", [
		PredOpTrait<"both first and second holds",
							  AllOf<[CPred<"first">, CPred<"second">]>>]> {
}

// CHECK-LABEL: OpB::verify
// CHECK: if (!(((first)) && ((second))))

def OpC : NS_Op<"op_for_TCopVTEtIs", [
    PredOpTrait<"first operand has i32 element type",
                TCopVTEtIs<0, I32>>]> {
  let arguments = (ins Tensor:$x);
}

// CHECK-LABEL: OpC::verify
// CHECK: if (!((((*this->getOperation()).getNumOperands() > 0)) && (((*this->getOperation()).getOperand(0)->getType().isa<ShapedType>())) && (((*this->getOperation()).getOperand(0)->getType().cast<ShapedType>().getElementType().isInteger(32)))))


def OpD : NS_Op<"op_for_TCOpVTEtIsSameAs", [
    PredOpTrait<"first operand is a vector or tensor with the same "
                "elemental type as itself",
                TCopVTEtIsSameAs<0, 0>>]> {
  let arguments = (ins Tensor:$x);
}

// CHECK-LABEL: OpD::verify
// CHECK: if (!((((*this->getOperation()).getNumOperands() > std::max(0,0))) && (((*this->getOperation()).getOperand(0)->getType().isa<ShapedType>())) && (((*this->getOperation()).getOperand(0)->getType().isa<ShapedType>())) && (((*this->getOperation()).getOperand(0)->getType().cast<ShapedType>().getElementType() == (*this->getOperation()).getOperand(0)->getType().cast<ShapedType>().getElementType()))))
// CHECK-NEXT: return emitOpError("failed to verify that first operand is a vector or tensor with the same elemental type as itself");


def OpE : NS_Op<"op_for_TCresVTEtIsSameAsOp", [
    PredOpTrait<"first operand is a vector or tensor with the same "
                "elemental type as first result",
                TCresVTEtIsSameAsOp<0, 0>>]> {
  let arguments = (ins Tensor:$x);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: OpE::verify
// CHECK: if (!((((*this->getOperation()).getNumResults() > 0)) && (((*this->getOperation()).getNumOperands() > 0)) && (((*this->getOperation()).getResult(0)->getType().isa<ShapedType>())) && (((*this->getOperation()).getOperand(0)->getType().isa<ShapedType>())) && (((*this->getOperation()).getResult(0)->getType().cast<ShapedType>().getElementType() == (*this->getOperation()).getOperand(0)->getType().cast<ShapedType>().getElementType()))))
// CHECK-NEXT: return emitOpError("failed to verify that first operand is a vector or tensor with the same elemental type as first result");


def OpF : NS_Op<"op_for_int_min_val", []> {
  let arguments = (ins Confined<I32Attr, [IntMinValue<10>]>:$attr);
}

// CHECK-LABEL: OpF::verify()
// CHECK:       (tblgen_attr.cast<IntegerAttr>().getInt() >= 10)
// CHECK-SAME:    return emitOpError("attribute 'attr' failed to satisfy constraint: 32-bit integer attribute whose minimal value is 10");

def OpG : NS_Op<"op_for_arr_min_count", []> {
  let arguments = (ins Confined<ArrayAttr, [ArrayMinCount<8>]>:$attr);
}

// CHECK-LABEL: OpG::verify()
// CHECK:       (tblgen_attr.cast<ArrayAttr>().size() >= 8)
// CHECK-SAME:    return emitOpError("attribute 'attr' failed to satisfy constraint: array attribute with at least 8 elements");

def OpH : NS_Op<"op_for_arr_value_at_index", []> {
  let arguments = (ins Confined<ArrayAttr, [IntArrayNthElemEq<0, 8>]>:$attr);
}

// CHECK-LABEL: OpH::verify()
// CHECK: (((tblgen_attr.cast<ArrayAttr>().size() > 0)) && ((tblgen_attr.cast<ArrayAttr>().getValue()[0].cast<IntegerAttr>().getInt() == 8)))))
// CHECK-SAME:    return emitOpError("attribute 'attr' failed to satisfy constraint: array attribute whose 0-th element must be 8");

def OpI: NS_Op<"op_for_arr_min_value_at_index", []> {
  let arguments = (ins Confined<ArrayAttr, [IntArrayNthElemMinValue<0, 8>]>:$attr);
}

// CHECK-LABEL: OpI::verify()
// CHECK: (((tblgen_attr.cast<ArrayAttr>().size() > 0)) && ((tblgen_attr.cast<ArrayAttr>().getValue()[0].cast<IntegerAttr>().getInt() >= 8)))))
// CHECK-SAME:    return emitOpError("attribute 'attr' failed to satisfy constraint: array attribute whose 0-th element must be at least 8");

def OpJ: NS_Op<"op_for_TCopVTEtAreSameAt", [
                PredOpTrait<"operands indexed at 0, 2, 3 should all have "
                 "the same type", TCopVTEtAreSameAt<[0, 2, 3]>>]> {
  let arguments = (ins
    Tensor:$a,
    Tensor:$b,
    Tensor:$c,
    Tensor:$d,
    Tensor:$e
  );
}

// CHECK-LABEL: OpJ::verify()
// CHECK:      llvm::is_splat(mlir::functional::map(
// CHECK-SAME:   [this](unsigned i) { return this->getOperand(i)->getType().cast<ShapedType>().getElementType(); },
// CHECK-SAME:   llvm::ArrayRef<unsigned>({0, 2, 3})))
// CHECK:   return emitOpError("failed to verify that operands indexed at 0, 2, 3 should all have the same type");

def OpK : NS_Op<"op_for_AnyTensorOf", []> {
  let arguments = (ins AnyTensorOf<[F32, I32]>:$x);
}

// CHECK-LABEL: OpK::verify
// CHECK: if (!(((this->getOperation()->getOperand(0)->getType().isa<TensorType>())) && (((this->getOperation()->getOperand(0)->getType().cast<TensorType>().getElementType().isF32())) || ((this->getOperation()->getOperand(0)->getType().cast<TensorType>().getElementType().isInteger(32))))))
