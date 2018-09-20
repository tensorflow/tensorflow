// RUN: mlir-opt %s -convert-to-cfg | FileCheck %s

// CHECK-LABEL: cfgfunc @empty_cfg() {
mlfunc @empty() {
             // CHECK: bb0:
  return     // CHECK:  return
}            // CHECK: }
