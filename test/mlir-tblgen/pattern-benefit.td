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
def : Pat<(X_AddOp (X_AddOp $lhs, $rhs), $rhs), (Y_AddOp $lhs, $rhs, $rhs)>;

// CHECK-LABEL: struct GeneratedConvert0
// CHECK: GeneratedConvert0(MLIRContext *context) : RewritePattern("x.add", 2, context) {}

def : Pat<(X_AddOp $lhs, $rhs), (Z_AddOp $lhs), [(IfEqual $lhs, $rhs)], (addBenefit 100)>;

// CHECK-LABEL: struct GeneratedConvert1
// CHECK: GeneratedConvert1(MLIRContext *context) : RewritePattern("x.add", 101, context) {}
