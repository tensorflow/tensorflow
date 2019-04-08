// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def FirstConstraint : AttrConstraint<CPred<"FirstConstraint">,
  "first constraint">;
def SecondConstraint : AttrConstraint<CPred<"SecondConstraint">,
  "second constraint">;
def ThirdConstraint : AttrConstraint<CPred<"ThirdConstraint">,
  "third constraint">;

def OpA : Op<"op_a", []> {
  let arguments = (ins
    I32Attr:$attr
  );

  let results = (outs I32:$result);
}

def : Pat<
  (OpA AllAttrConstraintsOf<
    [FirstConstraint,
     SecondConstraint,
     ThirdConstraint]>:$more),
  (OpA $more)>;

// Test combining AttrConstraint during pattern match.
// ---

// CHECK-LABEL: struct GeneratedConvert0

// CHECK:      auto attr = op0->getAttrOfType<IntegerAttr>("attr");
// CHECK: if (!(((FirstConstraint)) && ((SecondConstraint)) && ((ThirdConstraint)))) return matchFailure();
// CHECK-NEXT: s.more = attr;

