// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def SomeAttr : Attr<CPred<"some-condition">, "some attribute kind"> {
  let storageType = "some-attr-kind";
  let returnType = "some-return-type";
  let convertFromStorage = "{0}.some-convert-from-storage()";
  let constBuilderCall = "some-const-builder-call({0}, {1})";
}

def AOp : Op<"a_op", []> {
  let arguments = (ins
      SomeAttr:$aAttr,
      DefaultValuedAttr<SomeAttr, "4.2">:$bAttr,
      OptionalAttr<SomeAttr>:$cAttr
  );
}

// CHECK-LABEL: AOp definitions

// Test getter methods
// ---

// CHECK:      some-return-type AOp::aAttr() {
// CHECK-NEXT:   auto attr = this->getAttr("aAttr").dyn_cast_or_null<some-attr-kind>();
// CHECK-NEXT:   return attr.some-convert-from-storage();

// CHECK:      some-return-type AOp::bAttr() {
// CHECK-NEXT:   auto attr = this->getAttr("bAttr").dyn_cast_or_null<some-attr-kind>();
// CHECK-NEXT:   if (!attr)
// CHECK-NEXT:       return some-const-builder-call(mlir::Builder(this->getContext()), 4.2).some-convert-from-storage();
// CHECK-NEXT:   return attr.some-convert-from-storage();

// CHECK:      Optional<some-return-type> AOp::cAttr() {
// CHECK-NEXT:   auto attr = this->getAttr("cAttr").dyn_cast_or_null<some-attr-kind>();
// CHECK-NEXT:   return attr ? Optional<some-return-type>(attr.some-convert-from-storage()) : (llvm::None);

// Test build methods
// ---

// CHECK:      void AOp::build(
// CHECK:        tblgen_state->addAttribute("aAttr", aAttr);
// CHECK:        tblgen_state->addAttribute("bAttr", bAttr);
// CHECK:        if (cAttr) {
// CHECK-NEXT:     tblgen_state->addAttribute("cAttr", cAttr);

// CHECK:      void AOp::build(
// CHECK-SAME:   ArrayRef<NamedAttribute> attributes
// CHECK:        for (const auto& pair : attributes)
// CHECK-NEXT:     tblgen_state->addAttribute(pair.first, pair.second);

def MixOperandsAndAttrs : Op<"mix_operands_and_attrs", []> {
  let arguments = (ins F32Attr:$attr, F32:$operand, F32Attr:$otherAttr, F32:$otherArg);
}

// CHECK-LABEL: MixOperandsAndAttrs definitions
// CHECK-DAG: Value *MixOperandsAndAttrs::operand()
// CHECK-DAG: Value *MixOperandsAndAttrs::otherArg()
// CHECK-DAG: void MixOperandsAndAttrs::build(Builder *, OperationState *tblgen_state, FloatAttr attr, Value *operand, FloatAttr otherAttr, Value *otherArg)
// CHECK-DAG: APFloat MixOperandsAndAttrs::attr()
// CHECK-DAG: APFloat MixOperandsAndAttrs::otherAttr()
