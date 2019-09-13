// RUN: mlir-opt %s -pass-pipeline='module(test-module-pass,func(test-function-pass)),func(test-function-pass)' -cse -pass-pipeline="func(canonicalize)" -verify-each=false -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -pass-pipeline='module(test-module-pass' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_1 %s
// RUN: not mlir-opt %s -pass-pipeline='module(test-module-pass))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_2 %s
// RUN: not mlir-opt %s -pass-pipeline='module()(' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_3 %s
// RUN: not mlir-opt %s -pass-pipeline=',' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_4 %s

// CHECK_ERROR_1: encountered unbalanced parentheses while parsing pipeline
// CHECK_ERROR_2: encountered extra closing ')' creating unbalanced parentheses while parsing pipeline
// CHECK_ERROR_3: expected ',' after parsing pipeline
// CHECK_ERROR_4: does not refer to a registered pass or pass pipeline

func @foo() {
  return
}

module {
  func @foo() {
    return
  }
}

// CHECK: Pipeline Collection : ['func', 'module']
// CHECK-NEXT:   'func' Pipeline
// CHECK-NEXT:     TestFunctionPass
// CHECK-NEXT:     CSE
// CHECK-NEXT:       DominanceInfo
// CHECK-NEXT:     Canonicalizer
// CHECK-NEXT:   'module' Pipeline
// CHECK-NEXT:     TestModulePass
// CHECK-NEXT:     'func' Pipeline
// CHECK-NEXT:       TestFunctionPass
