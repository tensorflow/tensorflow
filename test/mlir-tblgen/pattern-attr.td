// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def MoreConstraint : AttrConstraint<CPred<"MoreConstraint">, "more constraint">;

def OpA : NS_Op<"op_a", []> {
  let arguments = (ins
    I32Attr:$required_attr,
    OptionalAttr<I32Attr>:$optional_attr,
    DefaultValuedAttr<I32Attr, "42">:$default_valued_attr,
		I32Attr:$more_attr
  );

  let results = (outs I32:$result);
}

def : Pat<(OpA $required, $optional, $default_valued, MoreConstraint:$more),
          (OpA $required, $optional, $default_valued, $more)>;

// Test attribute capturing
// ---

// CHECK-LABEL: struct GeneratedConvert0

// CHECK:      auto attr = op0->getAttrOfType<IntegerAttr>("required_attr");
// CHECK-NEXT: if (!attr) return matchFailure();
// CHECK-NEXT: s.required = attr;

// CHECK:      auto attr = op0->getAttrOfType<IntegerAttr>("optional_attr");
// CHECK-NEXT: s.optional = attr;

// CHECK:      auto attr = op0->getAttrOfType<IntegerAttr>("default_valued_attr");
// CHECK-NEXT: if (!attr) attr = mlir::Builder(ctx).getIntegerAttr(mlir::Builder(ctx).getIntegerType(32), 42);
// CHECK-NEXT: s.default_valued = attr;

// CHECK:      auto attr = op0->getAttrOfType<IntegerAttr>("more_attr");
// CHECK-NEXT: if (!attr) return matchFailure();
// CHECK-NEXT: if (!((MoreConstraint))) return matchFailure();
// CHECK-NEXT: s.more = attr;

