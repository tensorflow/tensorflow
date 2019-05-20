// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/OpBase.td"

// Check using the dialect name as the namespace
def A_Dialect : Dialect {
  let name = "a";
}

def A_SomeOp : Op<A_Dialect, "some_op", []>;

// Check a single namespace
def B_Dialect : Dialect {
  let name = "b";
  let cppNamespace = "BNS";
}

// Check nested namespaces
def B_SomeOp : Op<B_Dialect, "some_op", []>;

def C_Dialect : Dialect {
  let name = "c";
  let cppNamespace = "::C::CC";
}

def C_SomeOp : Op<C_Dialect, "some_op", []>;

// Check no namespaces
def D_Dialect : Dialect {
  let name = "d";
  let cppNamespace = "";
}

def D_DSomeOp : Op<D_Dialect, "some_op", []>;

// DEF-LABEL: GET_OP_LIST
// DEF:      a::SomeOp
// DEF-NEXT: BNS::SomeOp
// DEF-NEXT: ::C::CC::SomeOp
// DEF-NEXT: DSomeOp

// DEF-LABEL: GET_OP_CLASSES
// DEF: a::SomeOp definitions
// DEF: BNS::SomeOp definitions
// DEF: ::C::CC::SomeOp definitions
// DEF: DSomeOp definitions

// DECL-LABEL: GET_OP_CLASSES
// DECL: a::SomeOp declarations
// DECL: BNS::SomeOp declarations
// DECL: ::C::CC::SomeOp declarations
// DECL: DSomeOp declarations
