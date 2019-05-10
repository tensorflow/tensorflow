// RUN: mlir-tblgen -gen-rewriters -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def IfEqual : Constraint<CPred<"<notused>">>;

def Test_Dialect : Dialect {
  let name = "x";
}
class NS_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

// Define ops to rewrite.
def U: Type<CPred<"true">, "U">;
def X_AddOp : NS_Op<"add"> {
  let arguments = (ins U, U);
}
def Y_AddOp : NS_Op<"add"> {
  let arguments = (ins U, U, U);
}
def Z_AddOp : NS_Op<"add"> {
  let arguments = (ins U);
}

// Define rewrite patterns.
def bena : Pat<(X_AddOp (X_AddOp $lhs, $rhs), $rhs), (Y_AddOp $lhs, $rhs, $rhs)>;

// CHECK-LABEL: struct bena
// CHECK: RewritePattern("x.add", 2, context) {}

def benb : Pat<(X_AddOp $lhs, $rhs), (Z_AddOp $lhs), [(IfEqual $lhs, $rhs)], (addBenefit 100)>;

// CHECK-LABEL: struct benb
// CHECK: RewritePattern("x.add", 101, context) {}
