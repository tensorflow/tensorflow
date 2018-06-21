// TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
// statements (perhaps through using lit config substitutions).
//
// RUN: %S/../../mlir-opt --help | FileCheck --check-prefix=CHECKHELP %s
// RUN: %S/../../mlir-opt %s -o - | FileCheck %s
//
// CHECKHELP: OVERVIEW: MLIR modular optimizer driver


// Right now the input is completely ignored.
extfunc @foo()
extfunc @bar()

// CHECK: extfunc @foo()
// CHECK: extfunc @bar()
