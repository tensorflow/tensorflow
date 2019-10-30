// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include %s 2>&1 | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

def OpA : Op<Test_Dialect, "a"> {
  let arguments = (ins I32Attr:$attr);
}
def OpB : Op<Test_Dialect, "b"> {
  let arguments = (ins I32Attr:$attr);
}

def : Pat<(OpA $attr), (OpB $attr),
          // CHECK: operands to additional constraints can only be symbol references
          [(Constraint<CPred<"$0->getValue() == $1">> $attr, 42)]>;
