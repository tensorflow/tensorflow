// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF
// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s --check-prefix=PAT

include "mlir/IR/OpBase.td"

def NS_SomeEnum_A : EnumAttrCase<"A">;
def NS_SomeEnum_B : EnumAttrCase<"B">;

def NS_SomeEnum : EnumAttr<
  "SomeEnum", "some enum", [NS_SomeEnum_A, NS_SomeEnum_B]>;

def NS_OpA : Op<"op_a_with_enum_attr", []> {
  let arguments = (ins NS_SomeEnum:$attr);
  let results = (outs I32:$result);
}

// Test enum attribute getter method and verification
// ---

// DEF-LABEL: StringRef OpA::attr()
// DEF-NEXT:    auto attr = this->getAttr("attr").dyn_cast_or_null<StringAttr>();
// DEF-NEXT:    return attr.getValue();

// DEF-LABEL: OpA::verify()
// DEF:         auto tblgen_attr = this->getAttr("attr");
// DEF:         if (!(((tblgen_attr.isa<StringAttr>())) && (((tblgen_attr.cast<StringAttr>().getValue() == "A")) || ((tblgen_attr.cast<StringAttr>().getValue() == "B"))))) 
// DEF-SAME:      return emitOpError("attribute 'attr' failed to satisfy constraint: some enum");

def NS_OpB : Op<"op_b_with_enum_attr", []> {
  let arguments = (ins NS_SomeEnum:$attr);
  let results = (outs I32:$result);
}

def : Pat<(NS_OpA NS_SomeEnum_A:$attr), (NS_OpB NS_SomeEnum_B)>;

// Test enum attribute match and rewrite
// ---

// PAT-LABEL: struct GeneratedConvert0

// PAT:       PatternMatchResult match
// PAT:         auto attr = op0->getAttrOfType<StringAttr>("attr");
// PAT-NEXT:    if (!attr) return matchFailure();
// PAT-NEXT:    if (!((attr.cast<StringAttr>().getValue() == "A"))) return matchFailure();

// PAT:       void rewrite
// PAT:         auto vOpB0 = rewriter.create<NS::OpB>(loc,
// PAT-NEXT:        rewriter.getStringAttr("B")
// PAT-NEXT:    );

def NS_SomeEnum_Array : TypedArrayAttrBase<NS_SomeEnum, "SomeEnum array">;

def NS_OpC : Op<"op_b_with_enum_array_attr", []> {
  let arguments = (ins NS_SomeEnum_Array:$attr);
  let results = (outs I32:$result);
}

// Test enum array attribute verification
// ---

// DEF-LABEL: OpC::verify()
// DEF: [](Attribute attr) { return ((attr.isa<StringAttr>())) && (((attr.cast<StringAttr>().getValue() == "A")) || ((attr.cast<StringAttr>().getValue() == "B"))); }
